# Evaluation

## Offline checkpoint evaluation

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

Use `--policy prism_diffusion` for the diffusion interface, with its corresponding
checkpoint. Replace the example path with an actual saved model directory.
Keep the checkpoint's processor configuration, statistics and dataset split.

Offline evaluation compares predicted actions with recorded demonstrations.
An action error or thresholded proxy is not a robot task success rate. A held-out
claim requires evaluation episodes that were excluded from training. The helper
uses available split artifacts and supports explicit split selection; consult
`python scripts/eval_policy.py --help` for the current flags.

## Simulation

The environment adapters live under `scripts/env/`. Their simulator packages,
assets and recorded demonstrations are external requirements. Run an environment
independently before evaluating a model in it. Record task names, rollout seeds,
episode count and termination rules together with the generated summary.

The current RoboCasa rollout adapter supports **TRACE ACT only**. TRACE Diffusion
is supported for training and offline checkpoint evaluation; a RoboCasa online
diffusion adapter is not included. The public source checks do not run licensed
or separately downloaded simulator assets.

## Streaming inference

TRACE resets memory at episode boundaries. During a rollout it updates the memory
from the executed state and current observation, never from future frames.
`examples/smoke_trace.py` shows the model/data interface; focused tests compare a
causal prefix pass with online updates and check episode reset behavior.

The public source does not include the lab's ROS topic wiring, robot-specific
calibration or motion-command scripts. Real-robot evaluation needs an integration
that matches the training state/action layout, image preprocessing, signature
normalization and control timing. The source-only validation does not certify a
physical deployment or reproduce the paper's robot rollout scores.
