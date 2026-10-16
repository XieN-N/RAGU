"""Unit tests for ragu.decision.DecisionModel with a scripted LLM backend."""
import math
from collections.abc import Callable, Generator
from typing import Any

import pytest

from ragu.decision import DecisionModel, normalize_masses, shannon_entropy
from ragu.models.llm import LLM, LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.testing.openai_mock_server import OpenAIMockServer


class ScriptedLLM(LLM):
    """LLM whose option masses are computed from the last user message."""

    def __init__(self, script: Callable[[str, list[str]], dict[str, float]]):
        self.script = script
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, conversation, output_schema=str, **kwargs):  # type: ignore[override]
        raise AssertionError("decision models must not generate text")

    async def option_probabilities(self, conversation, options, **kwargs):  # type: ignore[override]
        self.calls.append({"conversation": conversation, "options": options, "kwargs": kwargs})
        return self.script(conversation[-1]["content"], options)


def _yes_if(*needles: str) -> Callable[[str, list[str]], dict[str, float]]:
    """P(Y)=0.9 for prompts containing any needle in the candidate block, else 0.1."""
    def script(prompt: str, options: list[str]) -> dict[str, float]:
        candidate = prompt.split("Candidate:")[1]
        hit = any(needle in candidate for needle in needles)
        return {"Y": 0.9 if hit else 0.1, "N": 0.1 if hit else 0.9}
    return script


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_normalize_masses_renormalizes() -> None:
    distribution, degraded = normalize_masses({"Y": 0.3, "N": 0.1, "X": 0.5}, ["Y", "N"])
    assert distribution == pytest.approx({"Y": 0.75, "N": 0.25})
    assert degraded is False


def test_normalize_masses_uniform_fallback() -> None:
    distribution, degraded = normalize_masses({}, ["A", "B", "C", "D"])
    assert distribution == pytest.approx({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    assert degraded is True


def test_shannon_entropy() -> None:
    assert shannon_entropy({"A": 1.0, "B": 0.0}) == 0.0
    assert shannon_entropy({"A": 0.5, "B": 0.5}) == pytest.approx(math.log(2))


# ---------------------------------------------------------------------------
# a_decide
# ---------------------------------------------------------------------------

async def test_decide_maps_labels_back_to_options() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"A": 0.1, "B": 0.6, "C": 0.1})
    model = DecisionModel(llm)
    decision = await model.a_decide(
        "Is the information sufficient?", ["STOP", "CONTINUE", "UNSURE"], context="facts",
    )
    assert decision.choice == "CONTINUE"
    assert decision.probability == pytest.approx(0.75)
    assert sum(decision.distribution.values()) == pytest.approx(1.0)
    assert decision.degraded is False
    prompt = llm.calls[0]["conversation"][-1]["content"]
    assert "A) STOP" in prompt and "B) CONTINUE" in prompt and "facts" in prompt
    assert llm.calls[0]["options"] == ["A", "B", "C"]


@pytest.mark.parametrize("n_options", [1, 21])
async def test_decide_rejects_bad_option_count(n_options: int) -> None:
    model = DecisionModel(ScriptedLLM(lambda prompt, options: {}))
    with pytest.raises(ValueError, match="options"):
        await model.a_decide("q", [f"opt{i}" for i in range(n_options)])


async def test_decide_degraded_when_no_label_returned() -> None:
    model = DecisionModel(ScriptedLLM(lambda prompt, options: {"Q": 1.0}))
    decision = await model.a_decide("q", ["x", "y"])
    assert decision.degraded is True
    assert decision.probability == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# a_score_pointwise
# ---------------------------------------------------------------------------

async def test_pointwise_threshold_sort_and_top_k() -> None:
    llm = ScriptedLLM(_yes_if("Paris", "Seine"))
    model = DecisionModel(llm)
    candidates = ["Berlin is in Germany", "Paris is in France", "The Seine flows through Paris"]
    output = await model.a_score_pointwise("Where is Paris?", candidates, threshold=0.5, top_k=1)

    assert output.n_calls == 3
    assert [c.index for c in output.scored] == [1, 2, 0]  # ties keep input order
    assert output.scored[0].probability == pytest.approx(0.9)
    assert [c.passed_threshold for c in output.scored] == [True, True, False]
    assert output.selected == ["Paris is in France"]
    assert output.degraded_count == 0
    # shared prefix first, candidate last (prefix caching on vLLM)
    prompt = llm.calls[0]["conversation"][-1]["content"]
    assert prompt.index("Where is Paris?") < prompt.index("Candidate:")


async def test_pointwise_uses_format_fn_and_state_context() -> None:
    llm = ScriptedLLM(_yes_if("#2"))
    model = DecisionModel(llm)
    output = await model.a_score_pointwise(
        "q", [1, 2, 3], format_fn=lambda x: f"item #{x}", state_context="chain so far",
    )
    assert output.selected == [2]
    assert "chain so far" in llm.calls[0]["conversation"][-1]["content"]


async def test_pointwise_empty_candidates_make_no_calls() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"Y": 1.0})
    output = await DecisionModel(llm).a_score_pointwise("q", [])
    assert output.selected == [] and output.n_calls == 0 and llm.calls == []


# ---------------------------------------------------------------------------
# a_score_listwise
# ---------------------------------------------------------------------------

async def test_listwise_ranks_from_one_call() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"A": 0.05, "B": 0.6, "C": 0.3, "Z": 0.05})
    model = DecisionModel(llm)
    output = await model.a_score_listwise("q", ["a", "b", "c"], top_k=2, min_prob=0.04)
    assert output.n_calls == 1 and len(llm.calls) == 1
    assert llm.calls[0]["options"] == ["A", "B", "C", "Z"]
    assert output.selected == ["b", "c"]
    assert output.none_probability == pytest.approx(0.05)


async def test_listwise_none_option_rejects_everything() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"A": 0.2, "Z": 0.8})
    output = await DecisionModel(llm).a_score_listwise("q", ["a", "b"])
    assert output.selected == []
    assert output.none_probability == pytest.approx(0.8)


async def test_listwise_rejects_too_many_candidates() -> None:
    model = DecisionModel(ScriptedLLM(lambda prompt, options: {}))
    with pytest.raises(ValueError, match="prefilter"):
        await model.a_score_listwise("q", [str(i) for i in range(20)])


# ---------------------------------------------------------------------------
# token constraints
# ---------------------------------------------------------------------------

_FAKE_VOCAB = {"Y": [10], " Y": [11], "N": [20], " N": [21], "A": [30], " A": [31, 32]}


async def test_logit_bias_covers_both_whitespace_variants() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"Y": 1.0})
    model = DecisionModel(llm, constraint="logit_bias", bias=50)
    model._encode = lambda text: _FAKE_VOCAB[text]
    await model.a_score_pointwise("q", ["c"])
    assert llm.calls[0]["kwargs"]["logit_bias"] == {10: 50, 11: 50, 20: 50, 21: 50}
    assert llm.calls[0]["kwargs"]["top_logprobs"] == 20


async def test_allowed_token_ids_go_to_extra_body() -> None:
    llm = ScriptedLLM(lambda prompt, options: {"Y": 1.0})
    model = DecisionModel(llm, constraint="allowed_token_ids")
    model._encode = lambda text: _FAKE_VOCAB[text]
    await model.a_score_pointwise("q", ["c"])
    assert llm.calls[0]["kwargs"]["extra_body"] == {"allowed_token_ids": [10, 11, 20, 21]}


def test_label_must_be_single_token() -> None:
    model = DecisionModel(ScriptedLLM(lambda prompt, options: {}), constraint="logit_bias")
    model._encode = lambda text: [1, 2]
    with pytest.raises(ValueError, match="single token"):
        model.label_token_ids(["Y"])


def test_unknown_constraint_rejected() -> None:
    with pytest.raises(ValueError, match="constraint"):
        DecisionModel(ScriptedLLM(lambda prompt, options: {}), constraint="grammar")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# end to end through the HTTP mock
# ---------------------------------------------------------------------------

@pytest.fixture
def server() -> Generator[OpenAIMockServer, None, None]:
    srv = OpenAIMockServer()
    srv.start()
    yield srv
    srv.stop()


async def test_pointwise_through_mock_server(server: OpenAIMockServer) -> None:
    server.set_top_logprobs([("Y", -0.4), (" Y", -2.5), ("N", -2.0)])
    llm = LLMOpenAI(CachedAsyncOpenAI(base_url=server.base_url, api_key="mock"), "mock")
    output = await DecisionModel(llm).a_score_pointwise("q", ["a", "b"])
    yes = math.exp(-0.4) + math.exp(-2.5)
    expected = yes / (yes + math.exp(-2.0))
    assert [c.probability for c in output.scored] == pytest.approx([expected, expected])
    assert output.selected == ["a", "b"]
