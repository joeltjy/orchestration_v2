# orchestration_v2

[![PyPI version](https://img.shields.io/pypi/v/orchestration_v2.svg)](https://pypi.org/project/orchestration_v2/)
[![PyPI downloads](https://static.pepy.tech/badge/orchestration_v2/month)](https://pepy.tech/projects/orchestration_v2)

SymphonyNet orchestral MIDI to per-track AMT token sequences.

* [GitHub](https://github.com/joeltjy/orchestration_v2/) | [PyPI](https://pypi.org/project/orchestration_v2/) | [Documentation](https://joeltjy.github.io/orchestration_v2/)
* Created by [Joel Tan](https://github.com/joeltjy) | GitHub [@joeltjy](https://github.com/joeltjy) | PyPI [@joeltjy](https://pypi.org/user/joeltjy/)
* MIT License

## Features

* TODO

## Installation

```bash
uv add orchestration_v2
```

## Usage

```python
from orchestration_v2.symphonynet import midi_to_track_events

sequences = midi_to_track_events("piece.mid")  # one AMT token list per track
```

## Dataset notes

The corpus is the SymphonyNet classical set in `~/Desktop/symphonynet_classical`.

* Removed 2026-11-03: `bach_cantate_62_bwv-62_6_(c)icking-archive.mid` and `beethoven_biamonti_291_(c)orlandi.mid`.
  No track in either file has 50 or more notes, too few to measure timing or reduce meaningfully.
* Corrected 2026-11-03: 2,480 tracks in 199 files were quantized but shifted off the beat by a constant number of
  ticks per track (mostly Lucarelli, Pajares and Bakels; offsets up to 1/8 of a beat). Each such track was moved
  back onto the beat; durations are unchanged and no notes were moved past tick 0. Originals, the per-track shift
  log (`_offset_log.csv`) and the scripts are in `~/Desktop/symphonynet_classical_offset_backup/`. The 38 files in
  `to_analyze/` were not touched.
* Corrected 2026-11-04: a second pass using exact tick counts shifted 269 more tracks in 145 files (mostly 1 tick
  late, which the first pass's +-1.25-tick tolerance let through). Logged in the same `_offset_log.csv`
  ("exact-tick pass"); files not already backed up were backed up first.

## Documentation

Full documentation is available on
[GitHub Pages](https://joeltjy.github.io/orchestration_v2/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
documentation instructions.

## Author

orchestration_v2 was created in 2026 by Joel Tan.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
