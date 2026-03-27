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
    Actions (SAC): position_size in [-1.0, +1.0]
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
        max_episode_steps: int = 252,
    ):
        super().__init__()
        assert len(prices) == len(features) == len(signals)
        self.prices = prices.astype(np.float64)
        self.features = features.astype(np.float32)
        self.signals = signals.astype(np.float32)
        self.mode = mode
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.max_episode_steps = max_episode_steps

        # observation: features + [portfolio_value_norm, position, days_held, unrealized_pnl, drawdown]
        state_dim = features.shape[1] + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        if mode == "dqn":
            self.action_space = gym.spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        else:  # sac
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # state variables (set in reset)
        self._cash = 0.0
        self._position = 0.0        # number of shares (can be fractional)
        self._portfolio_value = 0.0
        self._peak_value = 0.0
        self._entry_price = 0.0
        self._days_held = 0
        self._prev_position_sign = 0  # for flip detection
        self._step_count = 0
        self._start_idx = 0
        self._current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random start point ensuring enough room for an episode
        max_start = len(self.prices) - self.max_episode_steps - 1
        if max_start < 1:
            max_start = 1
        self._start_idx = self.np_random.integers(0, max_start)
        self._current_idx = self._start_idx

        self._cash = self.initial_capital
        self._position = 0.0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._entry_price = 0.0
        self._days_held = 0
        self._prev_position_sign = 0
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        price = self.prices[self._current_idx]
        next_idx = self._current_idx + 1
        next_price = self.prices[next_idx]
        price_change = (next_price - price) / price

        old_portfolio = self._portfolio_value

        # --- execute action ---
        trade_cost = 0.0
        if self.mode == "dqn":
            trade_cost = self._execute_dqn_action(int(action), price)
        else:
            target_position_frac = float(np.clip(action, -1.0, 1.0))
            trade_cost = self._execute_sac_action(target_position_frac, price)

        # --- advance to next price ---
        self._current_idx = next_idx
        self._portfolio_value = self._cash + self._position * next_price

        # track peak for drawdown
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        # days held
        if self._position != 0.0:
            self._days_held += 1
        else:
            self._days_held = 0

        # portfolio return
        portfolio_return = (self._portfolio_value - old_portfolio) / old_portfolio if old_portfolio > 0 else 0.0

        # position flip detection
        current_sign = int(np.sign(self._position))
        position_flipped = (current_sign != 0 and current_sign != self._prev_position_sign and self._prev_position_sign != 0)
        self._prev_position_sign = current_sign if current_sign != 0 else self._prev_position_sign

        # reward
        reward = self._compute_reward(
            portfolio_return=portfolio_return,
            trade_cost=trade_cost,
            position_flipped=position_flipped,
        )

        self._step_count += 1

        # termination: bankrupt or episode end
        terminated = self._portfolio_value <= self.initial_capital * 0.5  # 50% loss = game over
        truncated = self._step_count >= self.max_episode_steps or self._current_idx >= len(self.prices) - 1

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _execute_dqn_action(self, action: int, price: float) -> float:
        """Execute discrete action. Returns trading cost incurred."""
        trade_cost = 0.0
        if action == 1:  # Buy
            if self._position <= 0:
                # close short if any, then go long with all cash
                if self._position < 0:
                    close_value = abs(self._position) * price
                    trade_cost += close_value * self.trading_cost
                    self._cash -= abs(self._position) * price + close_value * self.trading_cost
                    self._position = 0.0
                # buy with available cash
                affordable = self._cash / (price * (1 + self.trading_cost))
                if affordable > 0:
                    trade_cost += affordable * price * self.trading_cost
                    self._cash -= affordable * price * (1 + self.trading_cost)
                    self._position = affordable
                    self._entry_price = price
        elif action == 2:  # Sell
            if self._position > 0:
                # close long position
                proceeds = self._position * price
                trade_cost = proceeds * self.trading_cost
                self._cash += proceeds - trade_cost
                self._position = 0.0
                self._entry_price = 0.0
        # action == 0: Hold, do nothing
        return trade_cost

    def _execute_sac_action(self, target_frac: float, price: float) -> float:
        """Execute continuous action (target position fraction). Returns trading cost."""
        # target_frac in [-1, 1]: fraction of portfolio to hold in position
        # positive = long, negative = short (short disabled for simplicity, clamp to 0)
        target_frac = max(target_frac, 0.0)  # no shorting for now
        target_value = self._portfolio_value * target_frac
        target_shares = target_value / price if price > 0 else 0.0

        delta_shares = target_shares - self._position
        if abs(delta_shares) < 1e-8:
            return 0.0

        trade_value = abs(delta_shares) * price
        trade_cost = trade_value * self.trading_cost

        if delta_shares > 0:
            # buying
            max_buyable = self._cash / (price * (1 + self.trading_cost))
            actual_delta = min(delta_shares, max_buyable)
            cost = actual_delta * price * (1 + self.trading_cost)
            self._cash -= cost
            self._position += actual_delta
            trade_cost = actual_delta * price * self.trading_cost
        else:
            # selling
            actual_delta = min(abs(delta_shares), self._position)
            proceeds = actual_delta * price * (1 - self.trading_cost)
            self._cash += proceeds
            self._position -= actual_delta
            trade_cost = actual_delta * price * self.trading_cost

        if self._position > 0 and self._entry_price == 0.0:
            self._entry_price = price
        elif self._position == 0.0:
            self._entry_price = 0.0

        return trade_cost

    def _compute_reward(
        self,
        portfolio_return: float,
        trade_cost: float,
        position_flipped: bool,
    ) -> float:
        """Pro Trader reward shaping."""
        r = portfolio_return

        # trading cost penalty (normalized)
        if self._portfolio_value > 0:
            r -= trade_cost / self._portfolio_value

        # drawdown penalty: penalize drawdowns beyond 2%
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value if self._peak_value > 0 else 0.0
        r -= max(0.0, drawdown - 0.02)

        # holding bonus: small reward for staying in position
        r += 0.0001 * self._days_held

        # flip penalty: discourage frequent direction changes
        if position_flipped:
            r -= 0.001

        return r

    def _get_obs(self) -> np.ndarray:
        idx = self._current_idx
        features = self.features[idx]

        portfolio_norm = self._portfolio_value / self.initial_capital
        position_sign = float(np.sign(self._position))
        days_held_norm = self._days_held / 252.0

        unrealized_pnl = 0.0
        if self._position != 0.0 and self._entry_price > 0:
            unrealized_pnl = (self.prices[idx] - self._entry_price) / self._entry_price * np.sign(self._position)

        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value if self._peak_value > 0 else 0.0

        portfolio_state = np.array(
            [portfolio_norm, position_sign, days_held_norm, unrealized_pnl, drawdown],
            dtype=np.float32,
        )
        return np.concatenate([features, portfolio_state])

    def _get_info(self) -> dict:
        return {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "position": self._position,
            "days_held": self._days_held,
            "step": self._step_count,
            "price": self.prices[self._current_idx],
        }
