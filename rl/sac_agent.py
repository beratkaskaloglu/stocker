"""
rl/sac_agent.py
Soft Actor-Critic — continuous pozisyon boyutu optimizasyonu.
"""
from __future__ import annotations


class SACTradingAgent:
    """
    Algoritma: Soft Actor-Critic (stable-baselines3)
    Amaç: position_size ∈ [-1.0, +1.0] optimize etmek
          -1.0 = tam short, 0.0 = cash, +1.0 = tam long

    PSEUDO:
    1. __init__(env, config):
       a. from stable_baselines3 import SAC
       b. SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=1,
          )
    2. train(total_timesteps=5_000_000):
       a. model.learn(total_timesteps, callback=EvalCallback)
       b. EvalCallback: her 10k step'te val env'de değerlendir
       c. Best model'i kaydet: outputs/models/sac_{market}.zip
    3. predict(state) → float
       a. action, _ = model.predict(state, deterministic=True)
       b. Clamp: [-1, +1]
    4. load(path) / save(path)
    """

    def __init__(self, env, config: dict):
        self.env = env
        self.config = config
        self.model = None  # TODO: SACfrom stable_baselines3

    def train(self, total_timesteps: int = 5_000_000) -> None:
        # TODO: implement
        raise NotImplementedError

    def predict(self, state) -> float:
        # TODO: implement
        raise NotImplementedError
