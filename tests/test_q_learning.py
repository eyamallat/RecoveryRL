import numpy as np

from agents.q_learning import QLearningAgent
from environment import Incident, RecoveryEnv


def test_q_learning_update_uses_terminal_reward_without_bootstrapping():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=0.9,
        exploration_rate=0,
        seed=1,
    )
    state = np.array([0, 0, 0, 1, 0, 1])

    agent.update(state, 2, 10, state, terminated=True)

    assert agent.q_table[tuple(state)][2] == 5


def test_training_learns_from_recovery_episodes():
    env = RecoveryEnv()
    agent = QLearningAgent(seed=7, exploration_decay=0.9)

    rewards = agent.train(env, episodes=20)

    assert len(rewards) == 20
    assert all(np.isfinite(reward) for reward in rewards)
    assert all(len(state) == 6 for state in agent.q_table)


def test_agent_can_run_a_non_exploratory_episode():
    env = RecoveryEnv()
    agent = QLearningAgent(seed=2)
    observation, _ = env.reset(options={"incident": Incident.CACHE_FAILURE})

    action = agent.select_action(observation, explore=False)

    assert 0 <= action < env.action_space.n
