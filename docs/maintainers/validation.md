# Source-release validation

Validated on 2026-10-08. Checks below concern source behavior and packaging, not
paper-result reproduction. Use the repository root for the listed commands.

## Results

| Check | Result |
| --- | --- |
| Full focused test suite | 73 passed |
| Utility-only suite in a separate environment without the model stack | 33 passed |
| CPU example with real Signatory features | Passed; finite actions with shape `(2, 2, 2)` |
| Training and evaluation `--help` | Passed; only the two TRACE policy types are exposed |
| Ruff correctness checks | Passed |
| Direct runtime dependency resolution for Python 3.12 | Passed |
| Two policy wheels | Built successfully; isolated import succeeded |
| Wheel license contents | LICENSE and NOTICE included in both packages |
| Four-language README command/citation alignment | Checked automatically |
| Local document links and source hygiene | Checked with `scripts/check_release.py` |

The model checks used Linux, Python 3.12, PyTorch 2.9.1, torchvision 0.24.1,
LeRobot 0.5.0, Signatory 1.2.7, NumPy 2.2.6 and Transformers 5.4.0. The local video
stack has TorchCodec 0.9.1+cu128; the portable requirements specify 0.9.1 and still
require a wheel / FFmpeg combination compatible with the target machine.
Developer tools were installed in temporary environments rather than changing
the existing research environment.

## Preservation of the complete method

The release specializes the existing policies to the complete TRACE computation.
With matching synthetic inputs and fixed seeds, it was compared against the
original complete-method configuration in separate processes:

- **ACT:** all 198 state-dict keys, tensor shapes and initialized values matched;
  predicted actions, training loss and the gradients of 117 parameters matched
  exactly.
- **Diffusion:** all 203 state-dict keys and initialized values matched; training
  loss, gradients of 202 parameters and sampled actions matched exactly.

These comparisons establish equivalence for the exercised small configurations.
They are not an exhaustive proof for every sensor layout or a verification
against unavailable paper checkpoints. The public tests separately cover causal
padding, prefix/online consistency, episode reset, diffusion backpropagation,
configuration round trips and rejection of unsupported method configurations.

## Re-run

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
python examples/smoke_trace.py
```

A clean utility-only environment can instead install `requirements-dev.txt` and
`requirements-test.txt`, then run `python -m pytest --lightweight`.

## Not validated in this packaging pass

- A fresh installation of the entire GPU/video stack on another host.
- Long training runs, published checkpoint downloads or exact paper metrics.
- Real-robot execution or asset-dependent simulator rollouts.
- Runtime compatibility with LeRobot versions other than the pinned 0.5.0.

The dependency resolver succeeded, but resolving packages is distinct from
installing and exercising all of them on a clean machine.
