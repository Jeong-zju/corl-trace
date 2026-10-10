<div align="center">

<img src="docs/assets/trace-logo.svg" alt="TRACE: trajectory-routed causal memory" width="440">

## TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation

**CoRL 2026 · Main Conference**

**Zihao Li<sup>1,2</sup> · Ranpeng Qiu<sup>2</sup> · Yincong Chen<sup>2</sup> · Guoqiang Ren<sup>2</sup> · Weiming Zhi<sup>1</sup>**

<sup>1</sup> **The University of Sydney** &emsp; <sup>2</sup> **Zeno AI**

[![CI](https://github.com/Jeong-zju/corl-trace/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jeong-zju/corl-trace/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://jeong-zju.github.io/trace/assets/paper/trace-paper.pdf)
[![Project](https://img.shields.io/badge/Project-Webpage-4b7bec.svg)](https://jeong-zju.github.io/trace/)
[![Code](https://img.shields.io/badge/Code-TRACE-2b9348.svg)](https://github.com/Jeong-zju/corl-trace)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

📄 [Paper](https://jeong-zju.github.io/trace/assets/paper/trace-paper.pdf) |
💻 [Code](https://github.com/Jeong-zju/corl-trace) |
🌐 [Webpage](https://jeong-zju.github.io/trace/) |
📚 [Documentation](docs/installation.md)

**English** · [简体中文](docs/i18n/README.zh-CN.md) · [日本語](docs/i18n/README.ja.md) · [한국어](docs/i18n/README.ko.md)

---

<img src="docs/assets/teaser.png" alt="TRACE retains early evidence to resolve later ambiguous decisions" width="100%">

</div>

## ✨ News

- **[2026/10/08]** The main-method source code is released with TRACE ACT / Diffusion interfaces and English, Chinese, Japanese and Korean documentation.

---

## 🔍 Motivation

**Remember the cue. Act when it matters.** TRACE gives visuomotor policies a
fixed-size causal memory for tasks where an early cue disappears before a later
decision. Path signatures summarize the executed robot-state trajectory and
route the writing and retrieval of visual-state evidence. Lightweight adapters
connect that memory to ACT and Diffusion Policy.

## ⚙️ Core features

| Component | Role |
| --- | --- |
| **Trajectory routing** | Path and delta signatures summarize the executed robot-state history and address memory. |
| **Causal slot memory** | A fixed set of latent slots retains visual-state evidence after an early cue disappears. |
| **Two policy interfaces** | Lightweight adapters condition ACT and Diffusion Policy through their native interfaces. |
| **Runnable source** | Data tools, reference recipes, a CPU example and focused tests support getting started. |

---

## 📚 Contents

- [Release contents](#release-contents)
- [Installation & quick start](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Code structure](#code-structure)
- [Development](#development)
- [Citation](#citation)
- [License & acknowledgments](#license)

---

<a id="release-contents"></a>

## 📦 Release contents

- **The complete TRACE method**, with regression (ACT) and diffusion interfaces.
- Causal prefix datasets, signature caching, training and offline / simulation evaluation.
- A CPU example with synthetic observations, real path signatures and no pretrained downloads.
- Reference configurations, focused tests and four-language documentation.

This release contains the main method only. Ablation implementations, comparison
experiments and private robot deployment configurations are excluded. Datasets,
trained weights and exact per-run paper artifacts are **not bundled**; the
reference configurations are starting points, not a claim of exact paper-result
reproduction. See [release scope](docs/release-scope.md).

---

<a id="installation"></a>

## 🚀 Installation & quick start

Linux and Python 3.12 are the reference environment. Clone the repository and
run the commands below from its root. Install system FFmpeg and a C++ compiler
before building Signatory. GPU training needs a PyTorch / CUDA installation
compatible with your driver.

```bash
git clone https://github.com/Jeong-zju/corl-trace.git
cd corl-trace

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e depends/signatory --no-build-isolation
python examples/smoke_trace.py
```

The example uses a small, randomly initialized CPU model. It checks that the
complete memory path produces finite actions; it does not measure task success.
Detailed environment and build notes: [installation](docs/installation.md).

---

<a id="training"></a>

## 🏋️ Training

Prepare a local LeRobot v3 dataset with synchronized RGB observations, robot state
and actions. Replace `data/local/task` with your dataset directory. Preprocessing
writes a separate output dataset so the input remains available.

```bash
python data/process_dataset.py data/local/task \
  --output-dir data/local/task_trace --signature-depth 3

python scripts/train_policy.py --policy streaming_act \
  --defaults-path configs/trace_act.yaml --dataset data/local/task_trace

python scripts/train_policy.py --policy prism_diffusion \
  --defaults-path configs/trace_diffusion.yaml --dataset data/local/task_trace
```

The two configurations enable the complete TRACE computation, use a fixed seed
and keep experiment tracking disabled by default. See [data preparation](docs/data.md)
and [training](docs/training.md) for splits, normalization and checkpoint details.

---

<a id="evaluation"></a>

## 🧪 Evaluation

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

Replace the checkpoint path with the actual saved directory. Offline action
errors are not robot success rates. Simulation requires the corresponding
environment and assets; see [evaluation](docs/evaluation.md).

---

<a id="code-structure"></a>

## 🗂️ Code structure

| Path | Purpose |
| --- | --- |
| `policy/lerobot_policy_streaming_act/src/` | TRACE memory and ACT interface |
| `policy/lerobot_policy_prism_diffusion/src/` | TRACE memory and Diffusion Policy interface |
| `configs/` | Public main-method training recipes |
| `scripts/` | Training, evaluation and dataset utilities |
| `data/` | LeRobot conversion and path-signature preprocessing |
| `examples/` | Runnable CPU example |
| `tests/` | Main-method, data and release checks |
| `docs/` | Installation, method, training, evaluation and translations |
| `depends/signatory/` | Vendored signature implementation and upstream license |

`streaming_act` and `prism_diffusion` are the historical serialized policy names
for the two TRACE interfaces. They are retained to keep full-method checkpoint
parameter names stable. See [architecture](docs/architecture.md) for the mapping
and compatibility requirements.

---

<a id="development"></a>

## 🛠️ Development

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

For a small CPU-only utility check without the model stack, install
`requirements-test.txt` and run `python -m pytest --lightweight`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

---

<a id="citation"></a>

## 📄 Citation

```bibtex
@inproceedings{li2026trace,
  title     = {TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation},
  author    = {Li, Zihao and Qiu, Ranpeng and Chen, Yincong and Ren, Guoqiang and Zhi, Weiming},
  booktitle = {Conference on Robot Learning},
  year      = {2026},
  url       = {https://jeong-zju.github.io/trace/}
}
```

---

<a id="license"></a>

## 📜 License & acknowledgments

Original TRACE contributions are licensed under [Apache-2.0](LICENSE).
We build on LeRobot, ACT, Diffusion Policy and Signatory; their notices and
licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
External data and weights retain their own licenses.

---

<div align="center">

**TRACE** · Remember the cue. Act when it matters.

</div>
