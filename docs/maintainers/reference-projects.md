# Open-source reference projects

Reviewed on **2026-10-08**. These official repositories informed the organization
and documentation of the TRACE release. Dates below refer to a specific release
or documented update, not necessarily the project's initial publication.

| Project | Recent evidence | Applied to TRACE |
| --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot/releases) | v0.6.0: 2026-07-06; v0.6.1: 2026-08-03. Dates checked against the official GitHub release API. | Keep the existing LeRobot policy extension and checkpoint interfaces. Separate optional dependencies, examples, tests, and documentation. |
| [StarVLA](https://github.com/starVLA/starVLA) | README news: VLAct code/models on 2026-08-27; consolidated RoboCasa365 guide on 2026-09-05. | Provide one complete data → training → evaluation recipe and a small standalone model smoke example. Distinguish reproduction instructions from measured results. |
| [OpenETA](https://github.com/OpenMOSS/OpenETA) | README news: first public release on 2026-07-25; Stage 2 update on 2026-09-06. | Put language navigation and research links near the top. Keep the quick start short and link to detailed workflow documents. |
| [RPent](https://github.com/RLinf/RPent) | README news: first paper in 2026-07; simulator integrations in 2026-08; leaderboard and robot extensions in 2026-09. | Use separate language READMEs with matching structure, explicit installation choices, and concrete task/checkpoint arguments. |
| [UnifoLM-WLA](https://github.com/unitreerobotics/unifolm-wla) | README news: action-expert training code on 2026-09-20; base model and fine-tuning code on 2026-09-28. | State the availability of code, data, and checkpoints separately. Keep third-party provenance visible. |

## Scope of the adaptation

- **Keep the working research interfaces.** TRACE remains implemented in the
  `lerobot_policy_streaming_act` extension. The public paper name does not require
  renaming saved policy types, configuration fields, or checkpoint keys.
- **Adopt a consistent Python style incrementally.** LeRobot and RPent both keep
  Ruff configuration in `pyproject.toml`. TRACE can follow the surrounding
  LeRobot conventions while limiting formatting changes to maintained code;
  whole-repository reformatting would obscure the research changes.
- **Keep four language entry points synchronized.** English, Simplified Chinese,
  Japanese, and Korean READMEs share commands, filenames, release status, and
  navigation. Translate explanations rather than CLI names or configuration keys.
- **Keep the implementation small.** Larger reference projects also serve many
  models, robots, and simulators. Their service layers, dashboards, plugin
  registries, and distributed launchers are not prerequisites for releasing this
  paper's method.
- **Treat examples as examples.** The CPU smoke run checks that a small model can
  execute with synthetic inputs. It does not reproduce paper results, validate a
  checkpoint, or demonstrate robot control.

These are organizational references, not additional algorithmic dependencies.
Existing upstream source notices and licenses remain authoritative for any
derived code. Fresh-install reproduction and benchmark results should only be
claimed after the corresponding runs have been completed.
