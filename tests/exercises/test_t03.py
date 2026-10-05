import pytest

from lagrl.contracts import GenerationEvent, PolicySpan, SamplingSettings, Termination, TokenRole
from lagrl.rollout import assemble_trajectory
from lagrl.testing.fixtures import trajectory_fixture


def test_event_alignment_and_sampler_provenance() -> None:
    tool = trajectory_fixture().tool_events[0]
    first = GenerationEvent((20, 21), (-0.25, -0.75), 1)
    second = GenerationEvent((22,), (-0.5,), 2)
    trajectory = assemble_trajectory(
        "task",
        "group",
        "response",
        (10, 11),
        (first, tool, second),
        SamplingSettings(),
        0.0,
        Termination.COMPLETED,
    )
    assert trajectory.token_ids == (10, 11, 20, 21, 30, 22)
    assert trajectory.token_roles == (
        TokenRole.PROMPT,
        TokenRole.PROMPT,
        TokenRole.ACTION,
        TokenRole.ACTION,
        TokenRole.TOOL,
        TokenRole.ACTION,
    )
    assert trajectory.action_mask == (False, False, True, True, False, True)
    assert trajectory.behavior_log_probs == (None, None, -0.25, -0.75, None, -0.5)
    assert trajectory.policy_spans == (PolicySpan(2, 4, 1), PolicySpan(5, 6, 2))
    assert trajectory.tool_events == (tool,)
    assert trajectory.reward == 0.0
    assert first.behavior_log_probs == (-0.25, -0.75)


def test_infrastructure_failure_stays_distinct_from_zero_reward() -> None:
    trajectory = assemble_trajectory(
        "task",
        "group",
        "response",
        (10,),
        (GenerationEvent((20,), (-0.5,), 1),),
        SamplingSettings(),
        None,
        Termination.INFRA_FAILURE,
    )
    assert trajectory.reward is None
    assert trajectory.termination is Termination.INFRA_FAILURE


@pytest.mark.parametrize(
    "events", [(), (GenerationEvent((20, 21), (-0.5,), 1),), (GenerationEvent((20,), (-0.5,), -1),)]
)
def test_malformed_or_empty_action_sequence_rejected(events: tuple) -> None:
    with pytest.raises(ValueError):
        assemble_trajectory(
            "task",
            "group",
            "response",
            (10,),
            events,
            SamplingSettings(),
            0.0,
            Termination.COMPLETED,
        )


def test_failure_cannot_become_zero_reward() -> None:
    with pytest.raises(ValueError):
        assemble_trajectory(
            "task",
            "group",
            "response",
            (10,),
            (GenerationEvent((20,), (-0.5,), 1),),
            SamplingSettings(),
            0.0,
            Termination.INFRA_FAILURE,
        )
