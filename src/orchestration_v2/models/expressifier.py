"""Expressifier: bidirectional dense-AMT-medium score encoder + causal performance decoder.

* Encoder: the dense-AMT medium GPT-2 trunk (24 x 1024, 16 heads), LM head removed, causal mask disabled
  (``enable_bidirectional_gpt2_attention``) so every score token sees the whole chunk.
* Decoder: like ``iod_pretrained``'s combo decoder (pre-norm ``nn.TransformerDecoder``, GELU) but causal over
  the 4N performance tokens and cross-attending to ALL encoder states (no memory mask). Inputs are
  [BOS] + performance tokens shifted right; token embeddings and the output head start from the encoder's
  dense-AMT ``wte``.
* Each position predicts one field of a quad (onset, duration, note, velocity); logits outside that field's
  token range are masked, in training and in generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput

from orchestration_v2.constants import (
    AMT_DENSE_DURATION_OFFSET,
    AMT_DENSE_MAX_DURATION,
    AMT_DENSE_MAX_ONSET,
    AMT_DENSE_NOTE_OFFSET,
    AMT_DENSE_ONSET_OFFSET,
    AMT_DENSE_VELOCITY_OFFSET,
    EXPRESSIFIER_BOS,
    EXPRESSIFIER_DECODER_INPUT_VOCAB,
    EXPRESSIFIER_ENCODER_CHECKPOINT,
    EXPRESSIFIER_MAX_TOKENS,
    EXPRESSIFIER_TOKENS_PER_NOTE,
    EXPRESSIFIER_VOCAB_SIZE,
)
from orchestration_v2.constants import LABEL_IGNORE_INDEX
from orchestration_v2.models.gpt2_utils import enable_bidirectional_gpt2_attention

DECODER_LAYERS = 6
DECODER_DROPOUT = 0.1
MASKED_LOGIT = -1e4  # bf16/fp16-safe "minus infinity"
T = EXPRESSIFIER_TOKENS_PER_NOTE
FIELD_RANGES = (  # [lo, hi] token ids per quad slot; note = piano (instrument 0) only
    (AMT_DENSE_ONSET_OFFSET, AMT_DENSE_ONSET_OFFSET + AMT_DENSE_MAX_ONSET),
    (AMT_DENSE_DURATION_OFFSET, AMT_DENSE_DURATION_OFFSET + AMT_DENSE_MAX_DURATION),
    (AMT_DENSE_NOTE_OFFSET, AMT_DENSE_NOTE_OFFSET + 127),
    (AMT_DENSE_VELOCITY_OFFSET, AMT_DENSE_VELOCITY_OFFSET + 127),
)


@dataclass
class ExpressifierOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


def field_allowed_mask(vocab_size: int = EXPRESSIFIER_VOCAB_SIZE) -> torch.Tensor:
    m = torch.zeros(T, vocab_size, dtype=torch.bool)
    for f, (lo, hi) in enumerate(FIELD_RANGES):
        m[f, lo:hi + 1] = True
    return m


class ExpressifierModel(nn.Module):
    def __init__(self, encoder: nn.Module, *, decoder_layers: int = DECODER_LAYERS,
                 dropout: float = DECODER_DROPOUT, max_tokens: int = EXPRESSIFIER_MAX_TOKENS,
                 vocab_size: int = EXPRESSIFIER_VOCAB_SIZE, init_from_encoder_wte: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        enable_bidirectional_gpt2_attention(self.encoder)
        hidden = int(encoder.config.n_embd)
        n_head = int(encoder.config.n_head)
        self.hidden, self.vocab_size, self.max_tokens = hidden, int(vocab_size), int(max_tokens)
        self.decoder_layers, self.dropout = int(decoder_layers), float(dropout)
        self.dec_wte = nn.Embedding(EXPRESSIFIER_DECODER_INPUT_VOCAB, hidden)
        self.dec_wpe = nn.Embedding(self.max_tokens, hidden)
        self.dec_field = nn.Embedding(T, hidden)
        layer = nn.TransformerDecoderLayer(d_model=hidden, nhead=n_head, dim_feedforward=4 * hidden,
                                           dropout=self.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.decoder_layers)
        self.dec_ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.vocab_size, bias=False)
        self.register_buffer("field_mask", field_allowed_mask(self.vocab_size), persistent=False)
        if init_from_encoder_wte:
            src = encoder.get_input_embeddings().weight.data
            n = min(self.vocab_size, src.shape[0])
            self.dec_wte.weight.data[:n].copy_(src[:n])
            self.head.weight.data[:n].copy_(src[:n])
            nn.init.normal_(self.dec_wpe.weight, std=0.02)
            nn.init.normal_(self.dec_field.weight, std=0.02)
        self.config = encoder.config
        self.config.expressifier = True

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    def decoder_inputs(self, output_ids: torch.Tensor) -> torch.Tensor:
        """[BOS] + output tokens shifted right by one."""
        dec = torch.full_like(output_ids, EXPRESSIFIER_BOS)
        dec[:, 1:] = output_ids[:, :-1].clamp(min=0)
        return dec

    def decode_logits(self, memory: torch.Tensor, memory_pad: torch.Tensor | None, dec_in: torch.Tensor,
                      dec_pad: torch.Tensor | None) -> torch.Tensor:
        L = dec_in.shape[1]
        pos = torch.arange(L, device=dec_in.device)
        x = self.dec_wte(dec_in) + self.dec_wpe(pos)[None] + self.dec_field(pos % T)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=dec_pad,
                         memory_key_padding_mask=memory_pad)
        logits = self.head(self.dec_ln(h))
        allowed = self.field_mask[pos % T]  # [L, V]
        return logits.masked_fill(~allowed[None], MASKED_LOGIT)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                output_ids: torch.Tensor | None = None, labels: torch.Tensor | None = None,
                **kwargs: Any) -> ExpressifierOutput:
        del kwargs
        if output_ids is None:
            if labels is None:
                raise ValueError("forward needs output_ids (teacher forcing) or labels")
            output_ids = labels.clamp(min=0)
        pad = None if attention_mask is None else ~attention_mask.bool()
        memory = self.encode(input_ids, attention_mask)
        logits = self.decode_logits(memory, pad, self.decoder_inputs(output_ids), pad)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.float().reshape(-1, self.vocab_size), labels.reshape(-1),
                                               ignore_index=LABEL_IGNORE_INDEX)
        return ExpressifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, *,
                 temperature: float = 1.0, top_p: float | None = 0.95, copy_pitch: bool = True,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample performance tokens for each score chunk (4N each). ``copy_pitch`` forces the note token to
        the score's note (the performance of note i is the performance of score note i)."""
        self.eval()
        B, L = input_ids.shape
        pad = None if attention_mask is None else ~attention_mask.bool()
        memory = self.encode(input_ids, attention_mask)
        out = torch.zeros(B, L, dtype=torch.long, device=input_ids.device)
        for t in range(L):  # no KV cache: chunks are <= 512 tokens
            logits = self.decode_logits(memory, pad, self.decoder_inputs(out)[:, :t + 1],
                                        None if pad is None else pad[:, :t + 1])[:, t].float()
            if copy_pitch and t % T == 2:
                nxt = input_ids[:, t]
            elif temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / temperature, -1)
                if top_p is not None and top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    keep = sp.cumsum(-1) - sp <= top_p
                    probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            out[:, t] = nxt
        if attention_mask is not None:
            out = out.masked_fill(~attention_mask.bool(), 0)
        return out


# ------------------------------------------------------------------ build / save / load
def _load_encoder(path: str | Path, *, local_files_only: bool = True) -> nn.Module:
    enc = AutoModel.from_pretrained(str(path), local_files_only=local_files_only)  # GPT2Model (LM head dropped)
    if int(enc.config.n_positions) < EXPRESSIFIER_MAX_TOKENS:
        raise ValueError(f"encoder n_positions={enc.config.n_positions} < {EXPRESSIFIER_MAX_TOKENS}")
    if int(enc.config.vocab_size) < EXPRESSIFIER_VOCAB_SIZE:
        raise ValueError(f"encoder vocab_size={enc.config.vocab_size} < dense-AMT {EXPRESSIFIER_VOCAB_SIZE}")
    return enc


def load_expressifier_model(encoder_checkpoint: str | Path = EXPRESSIFIER_ENCODER_CHECKPOINT, *,
                            decoder_layers: int = DECODER_LAYERS, dropout: float = DECODER_DROPOUT,
                            local_files_only: bool = True) -> ExpressifierModel:
    """Fresh expressifier: pretrained dense-AMT medium encoder + newly initialised decoder."""
    return ExpressifierModel(_load_encoder(encoder_checkpoint, local_files_only=local_files_only),
                             decoder_layers=decoder_layers, dropout=dropout)


DECODER_STATE = ("dec_wte", "dec_wpe", "dec_field", "decoder", "dec_ln", "head")


def save_expressifier(model: ExpressifierModel | Any, path: str | Path) -> None:
    raw = model.module if hasattr(model, "module") else model
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    raw.encoder.save_pretrained(dest / "encoder")
    torch.save({k: getattr(raw, k).state_dict() for k in DECODER_STATE}, dest / "decoder.pt")
    (dest / "expressifier_config.json").write_text(json.dumps(
        dict(decoder_layers=raw.decoder_layers, dropout=raw.dropout, max_tokens=raw.max_tokens,
             vocab_size=raw.vocab_size)) + "\n")


def is_expressifier_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    return (p / "decoder.pt").is_file() and (p / "encoder").is_dir()


def load_expressifier_checkpoint(path: str | Path, *, device: str | torch.device = "cpu",
                                 train: bool = False) -> ExpressifierModel:
    dest = Path(path)
    cfg = json.loads((dest / "expressifier_config.json").read_text())
    model = ExpressifierModel(_load_encoder(dest / "encoder"), decoder_layers=int(cfg["decoder_layers"]),
                              dropout=float(cfg["dropout"]), max_tokens=int(cfg["max_tokens"]),
                              vocab_size=int(cfg["vocab_size"]), init_from_encoder_wte=False)
    blob = torch.load(dest / "decoder.pt", map_location="cpu", weights_only=True)
    for k in DECODER_STATE:
        getattr(model, k).load_state_dict(blob[k])
    model.to(device)
    model.train(train)
    return model


__all__ = ["ExpressifierModel", "ExpressifierOutput", "FIELD_RANGES", "field_allowed_mask",
           "is_expressifier_checkpoint", "load_expressifier_checkpoint", "load_expressifier_model",
           "save_expressifier"]
