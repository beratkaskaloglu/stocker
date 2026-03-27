"""
rl/dqn_agent.py
Deep Q-Network — discrete buy/sell/hold kararı.
"""
from __future__ import annotations


class DQNTradingAgent:
    """
    Algoritma: DQN (stable-baselines3)
    Amaç: {0: Hold, 1: Buy, 2: Sell} kararını optimize etmek

    PSEUDO:
    1. __init__(env, config):
       a. from stable_baselines3 import DQN
       b. DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=256,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
          )
    2. train(total_timesteps=2_000_000):
       a. model.learn(total_timesteps, callback=EvalCallback)
       b. Best model'i kaydet: outputs/models/dqn_{market}.zip
    3. predict(state) → int (0, 1, veya 2)
    """

    def __init__(self, env, config: dict):
        self.env = env
        self.config = config
        self.model = None  # TODO: DQN from stable_baselines3

    def train(self, total_timesteps: int = 2_000_000) -> None:
        # TODO: implement
        raise NotImplementedError

    def predict(self, state) -> int:
        # TODO: implement
        raise NotImplementedError
