"""Tabular Q-Learning agent for discrete RecoveryRL observations."""

from collections import defaultdict

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_actions=6,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        min_exploration_rate=0.05,
        exploration_decay=0.995,
        seed=None,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= min_exploration_rate <= exploration_rate <= 1:
            raise ValueError(
                "exploration rates must satisfy 0 <= minimum <= initial <= 1"
            )
        if not 0 < exploration_decay <= 1:
            raise ValueError("exploration_decay must be in (0, 1]")

        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def select_action(self, observation, *, explore=True):
        values = self._values(observation)
        if explore and self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(values))

    def update(self, observation, action, reward, next_observation, terminated):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Invalid action: {action!r}")

        current_values = self._values(observation)
        next_values = self._values(next_observation)
        bootstrap_value = 0.0 if terminated else float(np.max(next_values))
        target = reward + self.discount_factor * bootstrap_value
        current_values[action] += self.learning_rate * (
            target - current_values[action]
        )

    def train(self, env, episodes):
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        episode_rewards = []
        for _ in range(episodes):
            observation, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.select_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                self.update(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                )
                observation = next_observation
                total_reward += reward

            episode_rewards.append(total_reward)
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )

        return episode_rewards

    def _values(self, observation):
        state = tuple(int(value) for value in observation)
        if not state:
            raise ValueError("observation must contain at least one value")
        return self.q_table[state]