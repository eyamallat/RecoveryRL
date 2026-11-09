import gymnasium as gym
import numpy as np
import pytest

from environment import Incident, RecoveryAction, RecoveryEnv


def test_observation_space_has_six_variables_and_324_states():
    env = RecoveryEnv()

    assert env.observation_space.nvec.tolist() == [3, 3, 3, 2, 3, 2]
    assert int(np.prod(env.observation_space.nvec)) == 324
    assert env.action_space.n == 6


@pytest.mark.parametrize(
    ("incident", "expected"),
    [
        (Incident.API_CPU_OVERLOAD, [2, 2, 0, 1, 0, 1]),
        (Incident.DATABASE_CONNECTION_EXHAUSTION, [0, 2, 2, 1, 0, 1]),
        (Incident.DATABASE_OUTAGE, [0, 2, 0, 0, 0, 1]),
        (Incident.CACHE_FAILURE, [0, 2, 0, 1, 2, 0]),
    ],
)
def test_reset_injects_incident_and_propagates_dependencies(incident, expected):
    env = RecoveryEnv()

    observation, info = env.reset(seed=5, options={"incident": incident})

    assert observation.tolist() == expected
    assert info["incident"] == incident
    assert env.observation_space.contains(observation)


def test_allocating_api_resources_reduces_overload_and_ends_episode():
    env = RecoveryEnv()
    env.reset(options={"incident": Incident.API_CPU_OVERLOAD})

    observation, reward, terminated, truncated, _ = env.step(
        RecoveryAction.ALLOCATE_API_RESOURCES
    )

    assert observation.tolist() == [1, 1, 0, 1, 0, 1]
    assert reward == 0
    assert not terminated
    assert not truncated

    observation, reward, terminated, truncated, info = env.step(
        RecoveryAction.ALLOCATE_API_RESOURCES
    )

    assert observation.tolist() == [0, 0, 0, 1, 0, 1]
    assert reward == 21
    assert terminated
    assert not truncated
    assert info["recovered"]


def test_database_outage_requires_database_recovery_action():
    env = RecoveryEnv()
    env.reset(options={"incident": Incident.DATABASE_OUTAGE})

    observation, _, terminated, _, _ = env.step(RecoveryAction.WAIT)
    assert observation[3] == 0
    assert observation[1] == 2
    assert not terminated

    observation, _, terminated, _, _ = env.step(RecoveryAction.RESTART_DATABASE)
    assert observation.tolist() == [0, 0, 0, 1, 0, 1]
    assert terminated


def test_episode_limit_truncates_unrecovered_incident():
    env = RecoveryEnv(max_episode_steps=1)
    env.reset(options={"incident": Incident.API_CPU_OVERLOAD})

    _, _, terminated, truncated, _ = env.step(RecoveryAction.WAIT)

    assert not terminated
    assert truncated


def test_invalid_action_and_incident_are_rejected():
    env = RecoveryEnv()
    with pytest.raises(ValueError, match="Unknown incident"):
        env.reset(options={"incident": "unknown"})
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_environment_passes_gymnasium_checker():
    env = RecoveryEnv()
    gym.utils.env_checker.check_env(env)