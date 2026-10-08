import os
import numpy as np


def configure_headless_opengl_defaults() -> None:
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    if os.environ.get("MUJOCO_GL") == "egl" and (not os.environ.get("PYOPENGL_PLATFORM")):
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def patch_lerobot_metaworld_env() -> None:
    from gymnasium import spaces
    from lerobot.envs.metaworld import MetaworldEnv

    if getattr(MetaworldEnv, "_corl_environment_state_patch", False):
        return
    original_init = MetaworldEnv.__init__
    original_format_raw_obs = MetaworldEnv._format_raw_obs

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        raw_obs_space = getattr(self._env, "observation_space", None)
        raw_obs_shape = tuple(getattr(raw_obs_space, "shape", ()) or ())
        if not raw_obs_shape:
            return
        obs_spaces = getattr(self.observation_space, "spaces", None)
        if not isinstance(obs_spaces, dict) or "environment_state" in obs_spaces:
            return
        patched_spaces = dict(obs_spaces)
        patched_spaces["environment_state"] = spaces.Box(
            low=np.full(raw_obs_shape, -np.inf, dtype=np.float32),
            high=np.full(raw_obs_shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(patched_spaces)

    def patched_format_raw_obs(self, raw_obs):
        observation = original_format_raw_obs(self, raw_obs)
        if "environment_state" not in observation:
            observation = dict(observation)
            observation["environment_state"] = (
                np.asarray(raw_obs, dtype=np.float32).reshape(-1).copy()
            )
        return observation

    MetaworldEnv.__init__ = patched_init
    MetaworldEnv._format_raw_obs = patched_format_raw_obs
    MetaworldEnv._corl_environment_state_patch = True
