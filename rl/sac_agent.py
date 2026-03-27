"""
rl/sac_agent.py
Soft Actor-Critic — continuous pozisyon boyutu optimizasyonu.
"""
from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


class SACTradingAgent:
    """
    SAC agent for continuous position sizing: [-1.0, +1.0]
    -1.0 = full short, 0.0 = cash, +1.0 = full long
    """

    DEFAULT_HYPERPARAMS = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "train_freq": 1,
        "gradient_steps": 1,
    }

    def __init__(self, env, config: dict | None = None, eval_env=None):
        self.env = env
        self.eval_env = eval_env
        self.config = config or {}

        hp = {**self.DEFAULT_HYPERPARAMS, **self.config.get("hyperparams", {})}

        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=hp["learning_rate"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            tau=hp["tau"],
            ent_coef=hp["ent_coef"],
            train_freq=hp["train_freq"],
            gradient_steps=hp["gradient_steps"],
            verbose=1,
            device=self.config.get("device", "auto"),
        )

    def train(
        self,
        total_timesteps: int = 5_000_000,
        eval_freq: int = 10_000,
        save_dir: str = "outputs/models",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        callbacks = []
        if self.eval_env is not None:
            eval_cb = EvalCallback(
                self.eval_env,
                best_model_save_path=save_dir,
                log_path=save_dir,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=1,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
        )

    def predict(self, state) -> float:
        action, _ = self.model.predict(state, deterministic=True)
        return float(action.clip(-1.0, 1.0))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = SAC.load(path, env=self.env)
