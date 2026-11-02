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

## Documentation

Full documentation is available on
[GitHub Pages](https://joeltjy.github.io/orchestration_v2/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
documentation instructions.

## Author

orchestration_v2 was created in 2026 by Joel Tan.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
