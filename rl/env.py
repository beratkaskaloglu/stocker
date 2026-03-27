"""
rl/env.py
Custom Gymnasium trading environment.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class StockerTradingEnv(gym.Env):
    """
    State: [price_features, signal_confidence, sentiment,
            portfolio_value_norm, current_position, days_held,
            unrealized_pnl, drawdown]

    Actions (DQN): 0=Hold, 1=Buy, 2=Sell
    Actions (SAC): position_size ∈ [-1.0, +1.0]

    PSEUDO:
    1. __init__(mode='dqn' | 'sac', initial_capital=100_000):
       a. observation_space: Box(shape=(state_dim,))
       b. action_space:
          - DQN: Discrete(3)
          - SAC: Box(-1, 1, shape=(1,))
       c. Trading cost: 0.001 (komisyon + spread)
    2. reset() → observation
       a. Portföyü sıfırla
       b. Rastgele başlangıç noktası seç (train setinden)
       c. İlk state'i döndür
    3. step(action) → (obs, reward, terminated, truncated, info)
       a. Aksiyon uygula (buy/sell/hold)
       b. Trading cost düş
       c. Yeni fiyatı uygula
       d. Reward hesapla (Pro Trader)
       e. Yeni state oluştur
    4. _compute_reward(action, price_change, portfolio_return) → float
       Pro Trader Reward:
         r = portfolio_return
         r -= trade_cost_penalty   (işlem varsa)
         r -= max(0, drawdown - 0.02)  (drawdown cezası)
         r += 0.0001 * days_held   (uzun tutma bonusu)
         r -= 0.001 * position_flip  (sık değiştirme cezası)
    5. render(mode='human') → None
       a. Portföy değeri, pozisyon, sinyal grafiği
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        signals: np.ndarray,
        mode: str = "dqn",
        initial_capital: float = 100_000,
        trading_cost: float = 0.001,
    ):
        super().__init__()
        self.prices = prices
        self.features = features
        self.signals = signals
        self.mode = mode
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost

        state_dim = features.shape[1] + 5  # features + portfolio state
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        if mode == "dqn":
            self.action_space = gym.spaces.Discrete(3)
        else:  # sac
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # TODO: implement
        raise NotImplementedError

    def step(self, action):
        # TODO: implement
        raise NotImplementedError

    def _compute_reward(self, action, price_change: float) -> float:
        # TODO: Pro Trader reward shaping
        raise NotImplementedError
