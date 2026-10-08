# Release scope

This source distribution contains the complete TRACE method with ACT and
Diffusion Policy interfaces, dataset/signature preparation, training, evaluation,
a CPU example, reference recipes and focused tests.

It excludes component-removal and alternative-memory experiments, comparison
sweeps, internal research notes, experiment logs, datasets, weights and lab-specific
robot deployment configuration. The original research checkout is maintained
separately and is not part of this distribution.

| Item | Status in this source bundle |
| --- | --- |
| TRACE ACT / diffusion implementations | Included |
| Main-method configurations | Included; reference recipes |
| Signatory source and license | Included |
| Synthetic example and focused tests | Included; see validation record |
| Exact task-specific paper run configurations | Not bundled |
| Datasets and trained checkpoints | Not bundled; no new public download asserted |
| Real-robot ROS integration | Not bundled |
| Main-conference acceptance | CoRL 2026, as confirmed by the authors |
| Proceedings DOI / pages | Not asserted before publication metadata is available |

See [validation](maintainers/validation.md) for the checks performed on the source
release. Passing these checks does not reproduce physical task success or certify
the reference recipes as the paper's exact experiment settings.
