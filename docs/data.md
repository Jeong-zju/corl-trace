# Data preparation

TRACE uses LeRobot v3 datasets containing episode metadata, synchronized RGB
observations (`observation.images.*`), robot state (`observation.state`) and
actions (`action`). Camera names and state/action dimensions come from dataset
metadata; the synthetic example deliberately uses small dimensions.

No demonstrations or trained model weights are included in the source release.
Bring your own authorized dataset or obtain the data through the paper's project
channels. A dataset identifier in an old research configuration is not evidence
that the dataset is publicly downloadable.

## Path-signature cache

```bash
python data/process_dataset.py data/local/task \
  --output-dir data/local/task_trace --signature-depth 3
```

The script processes each episode causally using Signatory. It creates a
`.signature_cache` directory and updates metadata for `observation.path_signature`
and `observation.delta_signature`. The output option keeps the source dataset
separate; omitting it updates the source in place. The default cache storage
dtype is float16; choose `--signature-cache-dtype float32` when needed.

Use the same state-channel order, masks, signature depth and normalization at
training and inference. Standard depth-3 signatures of 17-dimensional state have
17 + 17² + 17³ = 5,219 channels, excluding the scalar coordinate. A custom robot
state produces a different signature dimension.

## Splits and normalization

Public recipes use `test_ratio: 0.2` with a fixed split seed. Keep episode-level
membership fixed across training and evaluation. Preserve `dataset_split.json`
alongside the training outputs. A dataset too small to form both partitions needs
an explicit split decision; do not label training episodes as held-out results.

The general preprocessing utility computes cache statistics over the dataset
being processed. For a strict train-only normalization protocol, fit statistics
on training episodes and use those same frozen statistics for validation and
inference. Do not assume that running the cache command on a mixed train/test
dataset automatically reproduces the paper's normalization protocol.

## Conversion tools

`data/convert_legacy_dataset_to_v30.py --help` describes legacy LeRobot conversion.
`data/convert_rosbag_to_lerobot.py --help` describes configurable ROS bag import;
robot-specific bag paths and production configurations are not distributed.
`data/upload_dataset_to_hf.py` is an optional explicit upload tool. Authenticate
through your local Hugging Face login or environment; never commit credentials.
