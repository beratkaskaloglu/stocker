"""
rl/dqn_agent.py
Deep Q-Network — discrete buy/sell/hold karari.
"""
from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback


class DQNTradingAgent:
    """
    DQN agent for discrete trading decisions: {0: Hold, 1: Buy, 2: Sell}
    """

    DEFAULT_HYPERPARAMS = {
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
    }

    def __init__(self, env, config: dict | None = None, eval_env=None):
        self.env = env
        self.eval_env = eval_env
        self.config = config or {}

        hp = {**self.DEFAULT_HYPERPARAMS, **self.config.get("hyperparams", {})}

        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=hp["learning_rate"],
            buffer_size=hp["buffer_size"],
            learning_starts=hp["learning_starts"],
            batch_size=hp["batch_size"],
            tau=hp["tau"],
            gamma=hp["gamma"],
            train_freq=hp["train_freq"],
            target_update_interval=hp["target_update_interval"],
            exploration_fraction=hp["exploration_fraction"],
            exploration_final_eps=hp["exploration_final_eps"],
            verbose=1,
            device=self.config.get("device", "auto"),
        )

    def train(
        self,
        total_timesteps: int = 2_000_000,
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

    def predict(self, state) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path, env=self.env)
