# Installation

The reference stack is Linux, Python 3.12, PyTorch 2.9.1, torchvision 0.24.1 and
LeRobot 0.5.0. The release keeps the research API version instead of upgrading
the policy backend as part of repository cleanup.

## Python environment

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e depends/signatory --no-build-isolation
python examples/smoke_trace.py
```

The repository contains **two** installable policy packages. `requirements.txt`
installs both in editable mode; `pip install .` is not the root installation
command. Keep the repository available while using editable installs.

Install FFmpeg (including shared libraries for video decoding) and a C++ compiler
with your system package manager. Signatory builds against the installed PyTorch,
which is why it is installed separately with build isolation disabled. If a
parallel build exhausts RAM, retry with `MAX_JOBS=1` before the Signatory command.

The CPU example does not download model weights or contact experiment trackers.
The initial package installation still needs internet access or a wheel cache.
Training uses larger models and real data and can initialize a pretrained visual
backbone; plan for those additional downloads and storage.

## CUDA and video decoding

Use PyTorch wheels compatible with your NVIDIA driver. Requirements describe
Python packages; they do not install a GPU driver. TorchCodec must match the
installed PyTorch and have access to compatible FFmpeg shared libraries. A
successful CPU example verifies neither CUDA nor video decoding on your dataset.

RoboCasa, Meta-World and robot assets are optional external installations. They
are unnecessary for the synthetic example or offline evaluation. Install and
validate the specific simulator before using its rollout evaluation route.

## Development environment

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python scripts/check_release.py
```

For helper-only CI, use `requirements-dev.txt` plus `requirements-test.txt` in a
clean environment and `python -m pytest --lightweight`. Full tests additionally
need the model stack and Signatory. See [validation](maintainers/validation.md)
for checks actually run on this release.
