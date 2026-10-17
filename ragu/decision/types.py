"""Result types returned by :mod:`ragu.decision` models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ScoredCandidate(Generic[T]):
    """
    One candidate with the decision model's probability for it.

    :param index: Position of the candidate in the input sequence.
    :param item: The candidate itself (relation, chunk, community, ...).
    :param probability: Normalized probability in ``[0, 1]``: ``P(Y)`` for
        pointwise scoring, the candidate's share of the first-token
        distribution for listwise scoring.
    :param passed_threshold: Whether the candidate passed the selection rule
        (never for a degraded candidate).
    :param degraded: ``True`` if none of the option tokens appeared in the
        returned ``top_logprobs`` and the probability is a uniform fallback.
    """
    index: int
    item: T
    probability: float
    passed_threshold: bool
    degraded: bool = False


@dataclass(frozen=True)
class DecisionOutput(Generic[T]):
    """
    Outcome of scoring a candidate list.

    :param selected: Candidates that passed the selection rule, best first,
        cut to ``top_k`` when one was given.
    :param scored: Every candidate, sorted by probability descending
        (ties keep input order).
    :param n_calls: Number of LLM requests spent.
    :param none_probability: Listwise only: probability of the "none of the
        candidates" option.
    :param prompt_chars: Total characters of the prompts sent (a size proxy for
        the input tokens spent).
    """
    selected: list[T] = field(default_factory=list)
    scored: list[ScoredCandidate[T]] = field(default_factory=list)
    n_calls: int = 0
    none_probability: float | None = None
    prompt_chars: int = 0

    @property
    def degraded_count(self) -> int:
        """Number of candidates whose probability is a fallback value."""
        return sum(1 for candidate in self.scored if candidate.degraded)


@dataclass(frozen=True)
class SingleDecision:
    """
    Outcome of choosing exactly one option from a closed list.

    :param choice: The most probable option (the option text, not its label).
    :param probability: Probability of ``choice``.
    :param distribution: ``option -> probability`` over all options, sums to 1.
    :param entropy: Shannon entropy of ``distribution`` in nats.
    :param degraded: ``True`` if no option token was returned and the
        distribution is a uniform fallback.
    """
    choice: str
    probability: float
    distribution: dict[str, float]
    entropy: float
    degraded: bool = False
