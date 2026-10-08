<div align="center">

# TRACE

### Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation

**CoRL 2026 · Main Conference**

Zihao Li · Ranpeng Qiu · Yincong Chen · Guoqiang Ren · Weiming Zhi  
The University of Sydney · Zeno AI

[Project](https://jeong-zju.github.io/trace/) · [Quick start](#quick-start) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md) · [Citation](#citation)

**English** · [简体中文](docs/i18n/README.zh-CN.md) · [日本語](docs/i18n/README.ja.md) · [한국어](docs/i18n/README.ko.md)

</div>

![TRACE retains early evidence for decisions made after the cue disappears.](docs/assets/teaser.png)

**Remember the cue. Act when it matters.** TRACE gives visuomotor policies a
fixed-size causal memory for tasks where an early cue disappears before a later
decision. Path signatures summarize the executed robot-state trajectory and
route the writing and retrieval of visual-state evidence. Lightweight adapters
connect that memory to ACT and Diffusion Policy.

## What is included

- **The complete TRACE method**, with regression (ACT) and diffusion interfaces.
- Causal prefix datasets, signature caching, training and offline / simulation evaluation.
- A CPU example with synthetic observations, real path signatures and no pretrained downloads.
- Reference configurations, focused tests and four-language documentation.

This release contains the main method only. Ablation implementations, comparison
experiments and private robot deployment configurations are excluded. Datasets,
trained weights and exact per-run paper artifacts are **not bundled**; the
reference configurations are starting points, not a claim of exact paper-result
reproduction. See [release scope](docs/release-scope.md).

## Quick start

Use Linux and Python 3.12. Run these commands from the extracted repository root.
Install system FFmpeg and a C++ compiler before building Signatory. For GPU
training, use a PyTorch / CUDA installation compatible with your driver.

```bash
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

## Train on your data

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

## Evaluate a checkpoint

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

Replace the checkpoint path with the actual saved directory. Offline action
errors are not robot success rates. Simulation requires the corresponding
environment and assets; see [evaluation](docs/evaluation.md).

## Code map

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

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

For a small CPU-only utility check without the model stack, install
`requirements-test.txt` and run `python -m pytest --lightweight`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Citation

```bibtex
@inproceedings{li2026trace,
  title     = {TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation},
  author    = {Li, Zihao and Qiu, Ranpeng and Chen, Yincong and Ren, Guoqiang and Zhi, Weiming},
  booktitle = {Conference on Robot Learning},
  year      = {2026},
  url       = {https://jeong-zju.github.io/trace/}
}
```

## License and acknowledgments

Original TRACE contributions are licensed under [Apache-2.0](LICENSE).
We build on LeRobot, ACT, Diffusion Policy and Signatory; their notices and
licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
External data and weights retain their own licenses.
