"""
Single-token decision models.

A decision model answers a closed question with exactly one output token and
reads the answer distribution from the first token's ``top_logprobs``
instead of parsing generated text. One request costs a prompt prefill plus
one decoded token, the answer can never fall outside the option set, and the
probabilities can be thresholded.

Three primitives are provided:

- :meth:`BaseDecisionModel.a_decide`: pick one option from a closed list
  (routing, stop/continue).
- :meth:`BaseDecisionModel.a_score_pointwise`: independent ``Y``/``N``
  relevance judgement per candidate, all requests issued concurrently.
- :meth:`BaseDecisionModel.a_score_listwise`: one request over up to 19
  lettered candidates; the first-token distribution is read as a ranking.
"""
from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar

from ragu.common.base import RaguGenerativeModule
from ragu.common.logger import logger
from ragu.common.prompts.messages import ChatMessages, render
from ragu.decision.types import DecisionOutput, ScoredCandidate, SingleDecision
from ragu.models.llm import LLM

T = TypeVar("T")

#: OpenAI caps ``top_logprobs`` at 20, so at most 20 labels can be scored at once.
MAX_TOP_LOGPROBS = 20
#: Labels for :meth:`a_decide` options.
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRST"
#: Labels for listwise candidates; ``NONE_LABEL`` takes the 20th slot.
LISTWISE_LABELS = "ABCDEFGHIJKLMNOPQRS"
NONE_LABEL = "Z"
YES_LABEL = "Y"
NO_LABEL = "N"

#: How option tokens are constrained on the backend:
#: ``none`` reads ``top_logprobs`` as is; ``logit_bias`` boosts the label
#: tokens (OpenAI); ``allowed_token_ids`` restricts decoding to them (vLLM).
Constraint = Literal["none", "logit_bias", "allowed_token_ids"]

_DECISION_PROMPTS = ["decision_choice", "decision_pointwise", "decision_listwise"]


def normalize_masses(
    masses: dict[str, float],
    labels: Sequence[str],
) -> tuple[dict[str, float], bool]:
    """
    Renormalize raw option masses into a distribution over ``labels``.

    :param masses: Raw ``label -> probability mass`` from the backend.
    :param labels: Labels the distribution is defined over.
    :return: ``(distribution, degraded)``; when no label has positive mass the
        distribution is uniform and ``degraded`` is ``True``.
    """
    clipped = {label: max(masses.get(label, 0.0), 0.0) for label in labels}
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}, True
    return {label: mass / total for label, mass in clipped.items()}, False


def shannon_entropy(distribution: dict[str, float]) -> float:
    """Shannon entropy of a distribution, in nats."""
    return -sum(p * math.log(p) for p in distribution.values() if p > 0.0)


class BaseDecisionModel(ABC):
    """Interface of single-token decision models."""

    @abstractmethod
    async def a_decide(
        self,
        question: str,
        options: Sequence[str],
        context: str = "",
    ) -> SingleDecision:
        """
        Choose exactly one option.

        :param question: Closed question to answer.
        :param options: 2 to 20 option texts.
        :param context: Optional text the question is about.
        :return: The chosen option with the full distribution.
        """

    @abstractmethod
    async def a_score_pointwise(
        self,
        query: str,
        candidates: Sequence[T],
        format_fn: Callable[[T], str] = str,
        state_context: str = "",
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> DecisionOutput[T]:
        """
        Judge every candidate independently with a ``Y``/``N`` question.

        :param query: User query.
        :param candidates: Items to judge.
        :param format_fn: Renders one candidate as prompt text.
        :param state_context: Facts gathered so far (e.g. a reasoning chain).
        :param threshold: Minimum ``P(Y)`` for a candidate to be selected.
        :param top_k: Optional cap on the number of selected candidates.
        :return: Scored candidates and the selection.
        """

    @abstractmethod
    async def a_score_listwise(
        self,
        query: str,
        candidates: Sequence[T],
        format_fn: Callable[[T], str] = str,
        state_context: str = "",
        top_k: int = 3,
        min_prob: float = 0.05,
        max_none_prob: float = 0.5,
    ) -> DecisionOutput[T]:
        """
        Rank up to 19 candidates with one request.

        :param query: User query.
        :param candidates: 1 to 19 items; prefilter longer lists.
        :param format_fn: Renders one candidate as prompt text.
        :param state_context: Facts gathered so far.
        :param top_k: Maximum number of selected candidates.
        :param min_prob: Minimum candidate probability to be selected.
        :param max_none_prob: Nothing is selected when the "none" option is at
            least this probable.
        :return: Scored candidates and the selection.
        """


class DecisionModel(RaguGenerativeModule, BaseDecisionModel):
    """
    Decision model on top of any :class:`~ragu.models.llm.LLM` that exposes
    :meth:`~ragu.models.llm.LLM.option_probabilities` (e.g. ``LLMOpenAI``
    against OpenAI or vLLM).

    The model can be the same instance that generates answers, or a separate
    one (for example a local vLLM model while answers come from an API).

    :param llm: LLM backend.
    :param constraint: How label tokens are constrained (see
        :data:`Constraint`). ``none`` needs no tokenizer and is enough when the
        prompt makes the labels the obvious first token; ``logit_bias`` and
        ``allowed_token_ids`` need token ids and load a tokenizer.
    :param tokenizer_backend: ``tiktoken`` or ``local`` (HuggingFace), used
        only when ``constraint`` is not ``none``.
    :param tokenizer_model: Model or encoding name for the tokenizer, e.g.
        ``gpt-4o-mini``, ``o200k_base`` or ``bond005/meno-lite-0.1``.
    :param top_logprobs: Number of first-token alternatives to request.
    :param bias: Logit bias added to label tokens when ``constraint`` is
        ``logit_bias``.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        constraint: Constraint = "none",
        tokenizer_backend: Literal["tiktoken", "local"] = "tiktoken",
        tokenizer_model: str = "gpt-4o-mini",
        top_logprobs: int = MAX_TOP_LOGPROBS,
        bias: int = 100,
    ):
        super().__init__(prompts=list(_DECISION_PROMPTS))
        if constraint not in ("none", "logit_bias", "allowed_token_ids"):
            raise ValueError(f"Unsupported constraint: {constraint!r}")
        self.llm = llm
        self.constraint: Constraint = constraint
        self.tokenizer_backend = tokenizer_backend
        self.tokenizer_model = tokenizer_model
        self.top_logprobs = top_logprobs
        self.bias = bias
        self._encode: Callable[[str], list[int]] | None = None
        self._token_ids_cache: dict[tuple[str, ...], list[int]] = {}

    # -- token constraints -------------------------------------------------

    def _load_encoder(self) -> Callable[[str], list[int]]:
        """Load the tokenizer's ``encode`` function on first use."""
        if self.tokenizer_backend == "tiktoken":
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(self.tokenizer_model)
            except KeyError:
                encoding = tiktoken.get_encoding(self.tokenizer_model)
            return encoding.encode
        if self.tokenizer_backend == "local":
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)  # type: ignore[no-untyped-call]
            return lambda text: list(tokenizer.encode(text, add_special_tokens=False))
        raise ValueError(f"Unsupported tokenizer_backend: {self.tokenizer_backend!r}")

    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        """
        Token ids of the labels, both bare and with a leading space.

        :param labels: Option labels.
        :return: Sorted unique token ids.
        :raises ValueError: If a label is not a single token in either form.
        """
        key = tuple(labels)
        if key in self._token_ids_cache:
            return self._token_ids_cache[key]
        if self._encode is None:
            self._encode = self._load_encoder()
        ids: set[int] = set()
        for label in labels:
            found = False
            for variant in (label, f" {label}"):
                encoded = self._encode(variant)
                if len(encoded) == 1:
                    ids.add(encoded[0])
                    found = True
            if not found:
                raise ValueError(
                    f"Label {label!r} is not a single token for tokenizer "
                    f"{self.tokenizer_model!r}"
                )
        result = sorted(ids)
        self._token_ids_cache[key] = result
        return result

    def _request_kwargs(self, labels: Sequence[str]) -> dict[str, Any]:
        """Backend kwargs for one option-probabilities request."""
        kwargs: dict[str, Any] = {"top_logprobs": self.top_logprobs}
        if self.constraint == "logit_bias":
            kwargs["logit_bias"] = {
                token_id: self.bias for token_id in self.label_token_ids(labels)
            }
        elif self.constraint == "allowed_token_ids":
            kwargs["extra_body"] = {"allowed_token_ids": self.label_token_ids(labels)}
        return kwargs

    async def _distributions(
        self,
        conversations: Sequence[ChatMessages],
        labels: Sequence[str],
    ) -> list[tuple[dict[str, float], bool]]:
        """Request option masses for every conversation concurrently and normalize them."""
        kwargs = self._request_kwargs(labels)
        masses = await asyncio.gather(*[
            self.llm.option_probabilities(
                conversation.to_openai(), list(labels), **kwargs
            )
            for conversation in conversations
        ])
        results = [normalize_masses(mass, labels) for mass in masses]
        degraded = sum(1 for _, is_degraded in results if is_degraded)
        if degraded:
            logger.warning(
                f"DecisionModel: {degraded}/{len(results)} requests returned none of "
                f"the labels {list(labels)} in top_logprobs; using a uniform fallback"
            )
        return results

    # -- primitives ----------------------------------------------------------

    async def a_decide(
        self,
        question: str,
        options: Sequence[str],
        context: str = "",
    ) -> SingleDecision:
        if not 2 <= len(options) <= len(CHOICE_LABELS):
            raise ValueError(
                f"a_decide needs 2..{len(CHOICE_LABELS)} options, got {len(options)}"
            )
        labels = CHOICE_LABELS[:len(options)]
        conversation = render(
            self.get_prompt("decision_choice").messages,
            question=question,
            context=context,
            options_block="\n".join(
                f"{label}) {option}" for label, option in zip(labels, options)
            ),
            labels=", ".join(labels),
        )[0]
        [(label_distribution, degraded)] = await self._distributions([conversation], labels)

        distribution = {
            option: label_distribution[label] for label, option in zip(labels, options)
        }
        choice = max(options, key=lambda option: distribution[option])
        return SingleDecision(
            choice=choice,
            probability=distribution[choice],
            distribution=distribution,
            entropy=shannon_entropy(distribution),
            degraded=degraded,
        )

    async def a_score_pointwise(
        self,
        query: str,
        candidates: Sequence[T],
        format_fn: Callable[[T], str] = str,
        state_context: str = "",
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> DecisionOutput[T]:
        if not candidates:
            return DecisionOutput()
        conversations = render(
            self.get_prompt("decision_pointwise").messages,
            query=query,
            state_context=state_context,
            candidate=[format_fn(candidate) for candidate in candidates],
        )
        results = await self._distributions(conversations, (YES_LABEL, NO_LABEL))

        scored = [
            ScoredCandidate(
                index=i,
                item=candidate,
                probability=distribution[YES_LABEL],
                passed_threshold=distribution[YES_LABEL] >= threshold,
                degraded=degraded,
            )
            for i, (candidate, (distribution, degraded)) in enumerate(zip(candidates, results))
        ]
        scored.sort(key=lambda candidate: candidate.probability, reverse=True)
        selected = [candidate.item for candidate in scored if candidate.passed_threshold]
        if top_k is not None:
            selected = selected[:top_k]
        return DecisionOutput(selected=selected, scored=scored, n_calls=len(conversations))

    async def a_score_listwise(
        self,
        query: str,
        candidates: Sequence[T],
        format_fn: Callable[[T], str] = str,
        state_context: str = "",
        top_k: int = 3,
        min_prob: float = 0.05,
        max_none_prob: float = 0.5,
    ) -> DecisionOutput[T]:
        if not candidates:
            return DecisionOutput()
        if len(candidates) > len(LISTWISE_LABELS):
            raise ValueError(
                f"a_score_listwise takes at most {len(LISTWISE_LABELS)} candidates, "
                f"got {len(candidates)}; prefilter them first"
            )
        labels = LISTWISE_LABELS[:len(candidates)]
        conversation = render(
            self.get_prompt("decision_listwise").messages,
            query=query,
            state_context=state_context,
            candidates_block="\n".join(
                f"{label}) {format_fn(candidate)}"
                for label, candidate in zip(labels, candidates)
            ),
            first_label=labels[0],
            last_label=labels[-1],
            none_label=NONE_LABEL,
        )[0]
        [(distribution, degraded)] = await self._distributions(
            [conversation], (*labels, NONE_LABEL)
        )

        none_probability = distribution[NONE_LABEL]
        accept = none_probability < max_none_prob
        scored = [
            ScoredCandidate(
                index=i,
                item=candidate,
                probability=distribution[label],
                passed_threshold=accept and distribution[label] >= min_prob,
                degraded=degraded,
            )
            for i, (label, candidate) in enumerate(zip(labels, candidates))
        ]
        scored.sort(key=lambda candidate: candidate.probability, reverse=True)
        selected = [candidate.item for candidate in scored if candidate.passed_threshold][:top_k]
        return DecisionOutput(
            selected=selected,
            scored=scored,
            n_calls=1,
            none_probability=none_probability,
        )
