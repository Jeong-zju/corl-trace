# Contributing

Use Python 3.12 and follow [installation](docs/installation.md). Keep changes
focused and include the command, seed, input dimensions and observed failure when
reporting a bug. Small synthetic examples are preferable to private datasets.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

The public repository maintains the complete TRACE method. Keep method topology
fixed; alternative-memory experiments and component-removal sweeps belong in
separate research code. Preserve serialized parameter names when changing model
organization and test both training-prefix and online inference behavior.

Use typed interfaces for new public functions, concise docstrings and descriptive
names. Match the surrounding file style. Do not reformat vendored dependencies or
mix a broad formatting change with an algorithm change.

Keep English, Simplified Chinese, Japanese and Korean READMEs aligned. Commands,
paths and citation blocks must agree across translations. Detailed procedures
live under `docs/` to avoid duplicating long guides.

Contributions to original TRACE code are made under Apache-2.0. Preserve upstream
copyright headers and update `THIRD_PARTY_NOTICES.md` for newly bundled code.
Do not include credentials, personal paths, datasets, model weights or private
robot configuration in a source change.
