"""DecisionSearchEngine: context chosen by a decision model."""
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import asyncio
import random

import numpy as np
import pytest

from ragu.common.prompts.default_models import QueryPlan, SubQuery
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.decision import (
    BaseDecisionModel,
    DecisionModel,
    DecisionOutput,
    ScoredCandidate,
    SingleDecision,
)
from ragu.search_engine.decision_search import (
    DecisionParams,
    DecisionSearchEngine,
    DecisionSearchRetrieve,
)
from ragu.search_engine.local_search import LocalSearchEngine, LocalSearchRetrieve
from ragu.search_engine.query_plan import QueryPlanEngine
from ragu.storage.types import EmbeddingHit, Point


class KeywordDecisionModel(BaseDecisionModel):
    """P(Y)=0.9 for candidates whose text contains a keyword, else 0.1."""

    def __init__(self, keywords: Sequence[str]):
        self.keywords = [k.lower() for k in keywords]
        self.judged: list[str] = []

    async def a_decide(self, question, options, context=""):  # type: ignore[override]
        return SingleDecision(choice=options[0], probability=1.0, distribution={options[0]: 1.0}, entropy=0.0)

    async def a_score_pointwise(  # type: ignore[override]
        self,
        query: str,
        candidates: Sequence[Any],
        format_fn: Callable[[Any], str] = str,
        state_context: str = "",
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> DecisionOutput[Any]:
        scored = []
        for i, candidate in enumerate(candidates):
            text = format_fn(candidate)
            self.judged.append(text)
            p = 0.9 if any(k in text.lower() for k in self.keywords) else 0.1
            scored.append(ScoredCandidate(index=i, item=candidate, probability=p, passed_threshold=p >= threshold))
        scored.sort(key=lambda c: c.probability, reverse=True)
        return DecisionOutput(
            selected=[c.item for c in scored if c.passed_threshold],
            scored=scored,
            n_calls=len(candidates),
        )

    async def a_score_listwise(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


def _plain_engine(real_kg) -> LocalSearchEngine:
    """LocalSearch over the same graph: what the seed neighbourhood must equal."""
    return LocalSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3)),
    )


def _mock_retriever(engine, retrieved) -> None:
    engine.retriever.query_entities = AsyncMock(return_value=[retrieved])
    engine.retriever.build_query_vectors = AsyncMock(
        return_value=[Point(dense_embedding=np.ones(3072, dtype=np.float32))]
    )


def _decision_engine(real_kg, decision_model, retrieved) -> DecisionSearchEngine:
    engine = DecisionSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3)),
        decision_model=decision_model,
    )
    _mock_retriever(engine, retrieved)
    return engine


async def _seeds(real_kg, kg_fixture_ids, n_seeds: int = 4):
    seeds = [e for e in await real_kg.index.get_nodes(kg_fixture_ids["entity_ids"][:n_seeds]) if e]
    return seeds, [EmbeddingHit(id=e.id, distance=1.0 - 0.1 * i) for i, e in enumerate(seeds)]


async def _engine_with_seeds(real_kg, kg_fixture_ids, decision_model) -> DecisionSearchEngine:
    return _decision_engine(real_kg, decision_model, await _seeds(real_kg, kg_fixture_ids))


async def _full_context(real_kg, kg_fixture_ids) -> LocalSearchRetrieve:
    plain = _plain_engine(real_kg)
    _mock_retriever(plain, await _seeds(real_kg, kg_fixture_ids))
    return await plain.search("query")


async def test_pointwise_prunes_to_selected_and_floor(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    full = await _full_context(real_kg, kg_fixture_ids)

    # nothing passes: every section shrinks to the floor
    selected = await engine.search("query", DecisionParams(select_floor=1))
    result = selected.result
    assert len(result.entities) == min(1, len(full.result.entities))
    assert len(result.relations) == min(1, len(full.result.relations))
    assert len(result.chunks) == min(1, len(full.result.chunks))
    selection = selected.metrics["selection"]
    assert selection["entities"]["pool"] == len(full.result.entities)
    assert selection["relations"]["passed"] == 0
    assert decision_model.judged  # the decision model was actually asked


async def test_pointwise_keeps_candidates_that_pass(real_kg, kg_fixture_ids) -> None:
    full = await _full_context(real_kg, kg_fixture_ids)
    target = full.result.entities[1]

    decision_model = KeywordDecisionModel([target.entity_name])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", DecisionParams(select_floor=0))
    assert selected.result.entities[0].id == target.id
    assert all(target.entity_name.lower() in f"{e.entity_name} {e.description}".lower()
               for e in selected.result.entities)


async def test_pool_limits_bound_judged_candidates(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", DecisionParams(
        select_pool_entities=2, select_pool_relations=1,
        select_pool_chunks=1, select_floor=5,
    ))
    selection = selected.metrics["selection"]
    assert selection["entities"]["pool"] <= 2
    assert selection["relations"]["pool"] <= 1
    assert selection["chunks"]["pool"] <= 1
    # the floor cannot exceed the pool
    assert len(selected.result.entities) <= 2


async def _single_seed(real_kg):
    """One endpoint of a fixture relation as the only seed; returns (retrieved, relation, other endpoint)."""
    edges = await real_kg.index.graph_backend.get_all_edges()
    if not edges:
        pytest.skip("fixture graph has no relations")
    relation = edges[0]
    seed, frontier_entity = await real_kg.index.get_nodes([relation.subject_id, relation.object_id])
    return ([seed], [EmbeddingHit(id=seed.id, distance=1.0)]), relation, frontier_entity


async def test_hops_expand_only_from_passed_relations(real_kg) -> None:
    retrieved, relation, frontier_entity = await _single_seed(real_kg)
    decision_model = KeywordDecisionModel([relation.description])
    engine = _decision_engine(real_kg, decision_model, retrieved)
    selected = await engine.search("query", DecisionParams(select_hops=2, select_floor=0))

    hops = selected.metrics["selection_hops"]
    assert hops and hops[0]["hop"] == 1 and hops[0]["frontier"] >= 1
    # the frontier entity was judged on hop 1 with the kept facts as state
    assert any(text.startswith(f"{frontier_entity.entity_name} (") for text in decision_model.judged)
    assert relation.id in {r.id for r in selected.result.relations}


async def test_hops_stop_when_nothing_passes(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    selected = await engine.search("query", DecisionParams(select_hops=3))
    assert selected.metrics["selection_hops"] == []


async def test_single_hop_reports_no_hop_metrics(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    selected = await engine.search("query", DecisionParams())
    assert "selection_hops" not in selected.metrics


async def test_keep_seed_context_keeps_base_and_appends_hops(real_kg) -> None:
    retrieved, relation, frontier_entity = await _single_seed(real_kg)
    plain = _plain_engine(real_kg)
    _mock_retriever(plain, (await _single_seed(real_kg))[0])  # hops extend the retrieved lists
    full = await plain.search("query")

    decision_model = KeywordDecisionModel([relation.description, frontier_entity.entity_name])
    engine = _decision_engine(real_kg, decision_model, retrieved)
    selected = await engine.search("query", DecisionParams(
        select_hops=2, select_keep_seed_context=True,
    ))
    result = selected.result
    n_e, n_r, n_c = len(full.result.entities), len(full.result.relations), len(full.result.chunks)
    # the seed context is untouched and comes first
    assert [e.id for e in result.entities[:n_e]] == [e.id for e in full.result.entities]
    assert [r.id for r in result.relations[:n_r]] == [r.id for r in full.result.relations]
    assert [c.id for c in result.chunks[:n_c]] == [c.id for c in full.result.chunks]
    # the frontier entity passed on hop 1 and was appended
    assert frontier_entity.id in {e.id for e in result.entities[n_e:]}
    usage = selected.metrics["decision_usage"]
    assert usage["calls"] > 0  # the test double reports no prompt size
    assert {"retrieve", "select", "hops"} <= set(selected.metrics["timing"])


async def test_query_records_generation_time(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    engine.llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["answer"]))
    [response] = await engine.batch_query(["query"])
    assert response.response == "answer"
    assert isinstance(response.retrieval, DecisionSearchRetrieve)
    assert {"retrieve", "select", "generate"} <= set(response.retrieval.metrics["timing"])


async def test_engine_requires_decision_model(real_kg) -> None:
    arguments = {
        "llm": SimpleNamespace(chat_completion=AsyncMock()),
        "knowledge_graph": real_kg,
        "embedder": SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3)),
    }
    with pytest.raises(TypeError, match="decision_model"):
        DecisionSearchEngine(**arguments)
    with pytest.raises(ValueError, match="decision_model"):
        DecisionSearchEngine(**arguments, decision_model=None)


@pytest.mark.parametrize(
    "option,value",
    [
        ("state", "chain"),
        ("next_query", "subquestion"),
        ("stop", "budget"),
        ("max_context_tokens", 3000),
        ("max_decisions", 150),
        ("min_probability", 0.1),
        ("tie_break", "cosine"),
    ],
)
def test_best_first_options_need_best_first(option, value) -> None:
    with pytest.raises(ValueError, match=option):
        DecisionParams(**{option: value})
    DecisionParams(policy="best_first", **{option: value})


def test_unknown_option_value_raises() -> None:
    with pytest.raises(ValueError, match="policy"):
        DecisionParams(policy="beam")
    with pytest.raises(TypeError):
        DecisionParams(prerank="cosine")


async def test_seed_context_equals_local_search(real_kg, kg_fixture_ids) -> None:
    full = await _full_context(real_kg, kg_fixture_ids)
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    kept = await engine.search("query", DecisionParams(select_keep_seed_context=True))
    assert kept.metrics["entities"] == full.metrics["entities"]
    # the whole seed context remains and renders as in LocalSearch
    assert kept.to_text() == full.to_text()


class _NoLabelsLLM:
    """Backend whose first token is never one of the option labels."""

    async def option_probabilities(self, conversation, options, **kwargs):
        return {}


async def test_degraded_candidates_do_not_pass_the_threshold(real_kg, kg_fixture_ids) -> None:
    decision_model = DecisionModel(_NoLabelsLLM())  # type: ignore[arg-type]
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    # The uniform fallback gives P(Y) = 0.5, above this threshold.
    selected = await engine.search("query", DecisionParams(select_threshold=0.1, select_floor=0))
    selection = selected.metrics["selection"]
    assert selection["entities"]["degraded"] == selection["entities"]["pool"] > 0
    assert all(section["passed"] == 0 for section in selection.values())
    assert selected.result.entities == []


async def test_answer_prompt_override_is_used(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    engine.llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["answer"]))
    engine.update_prompt("local_search", RAGUInstruction(
        messages=ChatMessages.from_messages([UserMessage(content="Terse: {{ query }}")]),
        pydantic_model=str,
    ))
    await engine.batch_query(["query"])
    [conversation] = engine.llm.batch_chat_completion.await_args.args[0]
    assert conversation[-1]["content"] == "Terse: query"


async def test_stream_query_yields_deltas_with_context(real_kg, kg_fixture_ids) -> None:
    async def stream(conversation):
        yield "an"
        yield "swer"

    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    engine.llm = SimpleNamespace(stream_chat_completion=stream)
    events = [event async for event in engine.stream_query("query")]
    assert "".join(event.delta for event in events) == "answer"
    assert isinstance(events[0].retrieval, DecisionSearchRetrieve)


async def test_query_plan_wraps_decision_engine(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    plan = QueryPlan(subqueries=[SubQuery(id="q1", query="subquery")])
    engine.llm = SimpleNamespace(batch_chat_completion=AsyncMock(side_effect=[[plan], ["answer"]]))
    response = await QueryPlanEngine(engine).query("query", DecisionParams(select_floor=1))
    assert response.response == "answer"
    assert isinstance(response.retrieval, DecisionSearchRetrieve)
    assert len(response.retrieval.result.entities) == 1
    assert decision_model.judged


async def test_best_first_expands_below_the_threshold(real_kg) -> None:
    retrieved, relation, frontier_entity = await _single_seed(real_kg)
    # nothing passes, yet the walk still goes through the most probable relation
    decision_model = KeywordDecisionModel([])
    engine = _decision_engine(real_kg, decision_model, retrieved)
    selected = await engine.search("query", DecisionParams(policy="best_first", stop="budget", min_probability=0.5))

    assert any(text.startswith(f"{frontier_entity.entity_name} (") for text in decision_model.judged)
    assert selected.metrics["selection_walk"]["rounds"] >= 1
    # min_probability still guards the context
    assert not (selected.result.entities or selected.result.relations or selected.result.chunks)


async def test_best_first_respects_the_decision_budget(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", DecisionParams(
        policy="best_first", stop="budget", max_decisions=5, select_beam=1,
        select_pool_entities=1, select_pool_relations=1, select_pool_chunks=1,
    ))
    # three seed decisions, then at most the remaining allowance per round
    assert selected.metrics["decision_usage"]["calls"] <= 5 + 1


async def test_best_first_context_stays_within_the_token_budget(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel(["a", "e", "o"])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    wide = await engine.search("query", DecisionParams(policy="best_first", stop="budget"))
    narrow = await engine.search("query", DecisionParams(policy="best_first", stop="budget", max_context_tokens=30))
    size = lambda retrieve: len(retrieve.result.entities) + len(retrieve.result.relations) + len(retrieve.result.chunks)
    assert size(wide) > size(narrow)


async def test_chain_state_shows_the_path(real_kg) -> None:
    retrieved, relation, _ = await _single_seed(real_kg)
    states: list[str] = []

    class Recording(KeywordDecisionModel):
        async def a_score_pointwise(self, query, candidates, format_fn=str, state_context="", threshold=0.5, top_k=None):
            states.append(state_context)
            return await super().a_score_pointwise(query, candidates, format_fn, state_context, threshold, top_k)

    engine = _decision_engine(real_kg, Recording([relation.description]), retrieved)
    await engine.search("query", DecisionParams(policy="best_first", state="chain", stop="budget"))
    assert any(relation.description in state and state.count("--[") == 1 for state in states)


async def test_seed_facts_add_relations_and_endpoints(real_kg) -> None:
    retrieved, relation, frontier_entity = await _single_seed(real_kg)
    seed = retrieved[0][0]
    all_edges = await real_kg.index.graph_backend.get_all_edges()
    other = next((e for e in all_edges if seed.id not in (e.subject_id, e.object_id)), None)
    if other is None:
        pytest.skip("fixture graph has no relation away from the seed")
    decision_model = KeywordDecisionModel([other.description])
    engine = _decision_engine(real_kg, decision_model, retrieved)
    engine.retriever.query_relations = AsyncMock(return_value=[([other], [EmbeddingHit(id=other.id, distance=1.0)])])
    selected = await engine.search("query", DecisionParams(seeds="entities+facts", select_floor=0))

    assert other.id in {r.id for r in selected.result.relations}
    assert selected.metrics["seed_extension"]["relations"] >= 1


async def test_seed_names_match_whole_words(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    entity = next(e for e in await real_kg.index.get_nodes(kg_fixture_ids["entity_ids"]) if e and len(e.entity_name) >= 4)
    named = await engine._entities_named_in(f"What about {entity.entity_name}?")
    assert entity.id in {e.id for e in named}
    assert not await engine._entities_named_in(f"prefix{entity.entity_name.replace(' ', '')}suffix qqq")


async def test_one_round_judges_neighbours_without_state(real_kg) -> None:
    retrieved, relation, frontier_entity = await _single_seed(real_kg)
    states: list[str] = []

    class Recording(KeywordDecisionModel):
        async def a_score_pointwise(self, query, candidates, format_fn=str, state_context="", threshold=0.5, top_k=None):
            states.append(state_context)
            return await super().a_score_pointwise(query, candidates, format_fn, state_context, threshold, top_k)

    decision_model = Recording([])
    engine = _decision_engine(real_kg, decision_model, retrieved)
    await engine.search("query", DecisionParams(policy="one_round"))
    assert any(text.startswith(f"{frontier_entity.entity_name} (") for text in decision_model.judged)
    assert not any(states)


@pytest.mark.parametrize("order", ["ranked", "chunks_first", "flat"])
async def test_context_order_puts_probable_items_first(real_kg, kg_fixture_ids, order) -> None:
    full = await _full_context(real_kg, kg_fixture_ids)
    target = full.result.relations[-1]
    decision_model = KeywordDecisionModel([target.description])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", DecisionParams(context_order=order, select_floor=3))
    text = selected.to_text()
    assert target.description in text
    others = [r for r in selected.result.relations if r.id != target.id and r.description not in text.split(target.description)[0]]
    assert text.index(target.description) < min((text.index(r.description) for r in others), default=len(text))


async def test_cosine_tie_break_orders_equal_probabilities(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    params = dict(policy="best_first", stop="budget", min_probability=0.0, context_order="flat")
    plain = await engine.search("query", DecisionParams(**params))
    tied = await engine.search("query", DecisionParams(tie_break="cosine", **params))
    assert plain.result.order_scores == {}
    result = tied.result
    kept = [*result.entities, *result.relations, *result.chunks]
    assert kept and all(item.id in result.order_scores for item in kept)
    # every candidate has the same P(Y), so only the cosine can order them
    assert len({result.probabilities[item.id] for item in kept}) == 1
    assert len({result.order_scores[item.id] for item in kept}) > 1
    rendered = tied.to_text()
    first, last = (max(kept, key=lambda i: result.order_scores[i.id]), min(kept, key=lambda i: result.order_scores[i.id]))
    def text(item):
        return getattr(item, "content", None) or getattr(item, "description", "")
    assert rendered.index(text(first)[:40]) <= rendered.index(text(last)[:40])


async def _walk_with_random_delays(real_kg, kg_fixture_ids, seed: int) -> tuple[list[str], str]:
    """One best-first walk where every storage call takes a random time."""
    rng = random.Random(seed)

    def delayed(original):
        async def wrapper(*args, **kwargs):
            await asyncio.sleep(rng.random() * 0.01)
            return await original(*args, **kwargs)
        return wrapper

    index = real_kg.index
    decision_model = KeywordDecisionModel(["a", "e", "o"])
    # One seed: its relations lead to several new nodes, expanded in the same round.
    engine = _decision_engine(real_kg, decision_model, await _seeds(real_kg, kg_fixture_ids, 1))
    # An edge every expanded node has: parallel branches reach it at the same time.
    edges = await index.graph_backend.get_all_edges_for_nodes(kg_fixture_ids["entity_ids"][:4])
    template = next(edge for node_edges in edges for edge in node_edges if edge is not None)
    shared = replace(template, id="shared-edge", description="an edge shared by the branches")
    original_edges = index.graph_backend.get_all_edges_for_nodes

    async def edges_with_shared(ids):
        return [[*found, shared] for found in await original_edges(ids)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(index, "get_nodes", delayed(index.get_nodes))
        patch.setattr(index.graph_backend, "get_all_edges_for_nodes", delayed(edges_with_shared))
        patch.setattr(index.chunks_kv_storage, "get_by_ids", delayed(index.chunks_kv_storage.get_by_ids))
        selected = await engine.search("query", DecisionParams(
            policy="best_first", stop="budget", min_probability=0.0, select_beam=3, context_order="flat",
        ))
    return decision_model.judged, selected.to_text()


async def test_best_first_does_not_depend_on_timing(real_kg, kg_fixture_ids) -> None:
    runs = [await _walk_with_random_delays(real_kg, kg_fixture_ids, seed) for seed in range(6)]
    assert all(any("an edge shared" in text for text in judged) for judged, _ in runs)
    for judged, _ in runs:
        assert len(judged) == len(set(judged)), "a candidate was judged twice"
    assert len({tuple(judged) for judged, _ in runs}) == 1
    assert len({text for _, text in runs}) == 1
