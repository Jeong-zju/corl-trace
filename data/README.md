# Data tools

See [data preparation](../docs/data.md) for dataset format, signature caches,
split handling and normalization requirements.

| Script | Purpose |
| --- | --- |
| `process_dataset.py` | Compute causal path/delta signatures with Signatory |
| `convert_legacy_dataset_to_v30.py` | Convert older LeRobot datasets to v3 |
| `convert_rosbag_to_lerobot.py` | Import ROS bags using an explicit local YAML config |
| `upload_dataset_to_hf.py` | Explicitly upload a prepared dataset to Hugging Face |

Run a script with `--help` for its arguments. Dataset directories, credentials
and robot-specific conversion configurations are excluded from the source release.
