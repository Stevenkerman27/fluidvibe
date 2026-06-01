# Copyright 2026 Shruti Mishra. All rights reserved.
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config

from environments.taylor_green import (
    _ALIGNMENT_TIMESCALE,
    _DIFFUSIVITY_ROTATIONAL,
    _DIFFUSIVITY_TRANSLATIONAL,
    _FLOW_SPEED,
    _MIN_FLOW_SPEED_THRESHOLD,
    _SWIMMER_SPEED,
    _TIMESTEP,
    TaylorGreenEnvironment,
)


class TaylorGreenContinuousEnvironment(TaylorGreenEnvironment):

    def __init__(
        self,
        dt: float = _TIMESTEP,
        swimmer_speed: float = _SWIMMER_SPEED,
        flow_speed: float = _FLOW_SPEED,
        alignment_timescale: float = _ALIGNMENT_TIMESCALE,
        diffusivity_rotational: float = _DIFFUSIVITY_ROTATIONAL,
        diffusivity_translational: float = _DIFFUSIVITY_TRANSLATIONAL,
        seed: Optional[int] = None,
        action_type: Optional[str] = None,
    ):
        """Initialise the environment, with continuous observations and continuous or discrete actions.

        Args:
            action_type: "discrete" ∈ {0, 1, 2, 3} or "continuous" ∈ [0, 2π]
        """
        super().__init__(
            dt=dt,
            swimmer_speed=swimmer_speed,
            flow_speed=flow_speed,
            alignment_timescale=alignment_timescale,
            diffusivity_rotational=diffusivity_rotational,
            diffusivity_translational=diffusivity_translational,
            seed=seed,
        )
        self.action_type = action_type
        if self.action_type:
            if self.action_type not in ["discrete", "continuous"]:
                raise ValueError(
                    f"Invalid action_type {self.action_type!r}. Expected 'discrete', 'continuous', or None."
                )

    def _get_observation(self):
        """
        Returns:
            np.ndarray: observation = [vorticity_scaled, sin_theta, cos_theta], all are continuous-valued.
        """

        if abs(self.u0) > _MIN_FLOW_SPEED_THRESHOLD:
            vorticity_scaled = self.flow_vorticity / self.u0
        else:
            vorticity_scaled = 0

        orientation = np.arctan2(self.swimming_velocity[1], self.swimming_velocity[0])
        
        return np.array([vorticity_scaled, np.sin(orientation), np.cos(orientation)])

    def get_preferred_orientation(self, action):
        """Transforms the action into a preferred swimmer orientation."""

        if self.action_type == "continuous":
            orientation_preferred = action
        else:
            orientation_preferred = action * np.pi / 2

        return orientation_preferred


class TaylorGreenGymWrapper(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        # Extract environment-specific arguments from kwargs
        env_kwargs = {
            "dt": kwargs.get("dt", _TIMESTEP),
            "swimmer_speed": kwargs.get("swimmer_speed", _SWIMMER_SPEED),
            "flow_speed": kwargs.get("flow_speed", _FLOW_SPEED),
            "alignment_timescale": kwargs.get("alignment_timescale", _ALIGNMENT_TIMESCALE),
            "diffusivity_rotational": kwargs.get("diffusivity_rotational", _DIFFUSIVITY_ROTATIONAL),
            "diffusivity_translational": kwargs.get("diffusivity_translational", _DIFFUSIVITY_TRANSLATIONAL),
            "seed": kwargs.get("seed", config.SEED),
            "action_type": kwargs.get("action_type", "discrete")
        }
        self.env = TaylorGreenContinuousEnvironment(**env_kwargs)
        
        # State: [vorticity_scaled, sin_theta, cos_theta]
        low = np.array([-np.inf, -1.0, -1.0], dtype=np.float32)
        high = np.array([np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        if self.env.action_type == "discrete":
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=0.0, high=2*np.pi, shape=(1,), dtype=np.float32)
            
        self.steps = 0
        self.max_steps = config.N_STEPS
        self.cumulative_y_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed=seed)
        
        obs = self.env.reset()
        self.steps = 0
        self.cumulative_y_dist = 0.0
        return obs.astype(np.float32), {}

    def step(self, action):
        old_y = self.env.swimmer_position[1]
        obs, reward = self.env.step(action)
        new_y = self.env.swimmer_position[1]
        dy = new_y - old_y
        self.cumulative_y_dist += dy
        
        self.steps += 1
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        info = {}
        if terminated or truncated:
            info["y_dist"] = self.cumulative_y_dist
            
        return obs.astype(np.float32), float(reward), terminated, truncated, info


# Register the environment
gym.envs.registration.register(
    id="TaylorGreen-v0",
    entry_point="environments.taylor_green_continuous:TaylorGreenGymWrapper",
)
