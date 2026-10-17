"""LocalSearchEngine with a decision-model selector (LocalParams.selector)."""
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ragu.decision import (
    BaseDecisionModel,
    DecisionOutput,
    ScoredCandidate,
    SingleDecision,
)
from ragu.search_engine.local_search import LocalParams, LocalSearchEngine
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


async def _engine_with_seeds(real_kg, kg_fixture_ids, decision_model, n_seeds: int = 4) -> LocalSearchEngine:
    engine = LocalSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3)),
        decision_model=decision_model,
    )
    seeds = [e for e in await real_kg.index.get_nodes(kg_fixture_ids["entity_ids"][:n_seeds]) if e]
    engine.retriever.query_entities = AsyncMock(return_value=[(
        seeds,
        [EmbeddingHit(id=e.id, distance=1.0 - 0.1 * i) for i, e in enumerate(seeds)],
    )])
    engine.retriever.build_query_vectors = AsyncMock(
        return_value=[Point(dense_embedding=np.ones(3072, dtype=np.float32))]
    )
    return engine


async def test_selector_none_leaves_context_untouched(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    baseline = await engine.search("query", LocalParams())
    assert decision_model.judged == []
    assert "selection" not in baseline.metrics


async def test_pointwise_prunes_to_selected_and_floor(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    full = await engine.search("query", LocalParams())

    # nothing passes: every section shrinks to the floor
    selected = await engine.search("query", LocalParams(selector="pointwise", select_floor=1))
    result = selected.result
    assert len(result.entities) == min(1, len(full.result.entities))
    assert len(result.relations) == min(1, len(full.result.relations))
    assert len(result.chunks) == min(1, len(full.result.chunks))
    selection = selected.metrics["selection"]
    assert selection["entities"]["pool"] == len(full.result.entities)
    assert selection["relations"]["passed"] == 0
    assert decision_model.judged  # the decision model was actually asked


async def test_pointwise_keeps_candidates_that_pass(real_kg, kg_fixture_ids) -> None:
    probe_engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    full = await probe_engine.search("query", LocalParams())
    target = full.result.entities[1]

    decision_model = KeywordDecisionModel([target.entity_name])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", LocalParams(selector="pointwise", select_floor=0))
    assert selected.result.entities[0].id == target.id
    assert all(target.entity_name.lower() in f"{e.entity_name} {e.description}".lower()
               for e in selected.result.entities)


async def test_pool_limits_bound_judged_candidates(real_kg, kg_fixture_ids) -> None:
    decision_model = KeywordDecisionModel([])
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, decision_model)
    selected = await engine.search("query", LocalParams(
        selector="pointwise", select_pool_entities=2, select_pool_relations=1,
        select_pool_chunks=1, select_floor=5,
    ))
    selection = selected.metrics["selection"]
    assert selection["entities"]["pool"] <= 2
    assert selection["relations"]["pool"] <= 1
    assert selection["chunks"]["pool"] <= 1
    # the floor cannot exceed the pool
    assert len(selected.result.entities) <= 2


async def test_cosine_selector_keeps_top_n_without_decision_model(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, None)
    full = await engine.search("query", LocalParams())
    selected = await engine.search("query", LocalParams(
        selector="cosine", select_top_entities=2, select_top_relations=1, select_top_chunks=1,
    ))
    result = selected.result
    # entities keep retrieval order; the pools are the same ones pointwise would judge
    assert [e.id for e in result.entities] == [e.id for e in full.result.entities[:2]]
    assert len(result.relations) == min(1, len(full.result.relations))
    assert len(result.chunks) == min(1, len(full.result.chunks))
    assert all(section["n_calls"] == 0 for section in selected.metrics["selection"].values())


async def _single_seed_engine(real_kg, decision_model):
    """Engine seeded with one endpoint of a fixture relation; returns (engine, relation, other endpoint)."""
    edges = await real_kg.index.graph_backend.get_all_edges()
    if not edges:
        pytest.skip("fixture graph has no relations")
    relation = edges[0]
    seed, frontier_entity = await real_kg.index.get_nodes([relation.subject_id, relation.object_id])
    engine = LocalSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=SimpleNamespace(embed_text=AsyncMock(return_value=[0.0] * 3)),
        decision_model=decision_model,
    )
    engine.retriever.query_entities = AsyncMock(return_value=[([seed], [EmbeddingHit(id=seed.id, distance=1.0)])])
    engine.retriever.build_query_vectors = AsyncMock(
        return_value=[Point(dense_embedding=np.ones(3072, dtype=np.float32))]
    )
    return engine, relation, frontier_entity


async def test_hops_expand_only_from_passed_relations(real_kg) -> None:
    _, relation, frontier_entity = await _single_seed_engine(real_kg, None)
    decision_model = KeywordDecisionModel([relation.description])
    engine, _, _ = await _single_seed_engine(real_kg, decision_model)
    selected = await engine.search("query", LocalParams(selector="pointwise", select_hops=2, select_floor=0))

    hops = selected.metrics["selection_hops"]
    assert hops and hops[0]["hop"] == 1 and hops[0]["frontier"] >= 1
    # the frontier entity was judged on hop 1 with the kept facts as state
    assert any(text.startswith(f"{frontier_entity.entity_name} (") for text in decision_model.judged)
    assert relation.id in {r.id for r in selected.result.relations}


async def test_hops_stop_when_nothing_passes(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    selected = await engine.search("query", LocalParams(selector="pointwise", select_hops=3))
    assert selected.metrics["selection_hops"] == []


async def test_single_hop_reports_no_hop_metrics(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, KeywordDecisionModel([]))
    selected = await engine.search("query", LocalParams(selector="pointwise"))
    assert "selection_hops" not in selected.metrics


async def test_keep_seed_context_keeps_base_and_appends_hops(real_kg) -> None:
    _, relation, frontier_entity = await _single_seed_engine(real_kg, None)
    plain, _, _ = await _single_seed_engine(real_kg, None)
    full = await plain.search("query", LocalParams())

    decision_model = KeywordDecisionModel([relation.description, frontier_entity.entity_name])
    engine, _, _ = await _single_seed_engine(real_kg, decision_model)
    selected = await engine.search("query", LocalParams(
        selector="pointwise", select_hops=2, select_keep_seed_context=True,
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


async def test_ppr_appends_new_chunks_from_similarity_seeds(real_kg) -> None:
    engine, _, _ = await _single_seed_engine(real_kg, None)
    full = await engine.search("query", LocalParams())
    with_ppr = await engine.search("query", LocalParams(ppr_chunks=5))
    base_ids = [c.id for c in full.result.chunks]
    ids = [c.id for c in with_ppr.result.chunks]
    assert ids[:len(base_ids)] == base_ids
    assert len(ids) == len(set(ids))  # no duplicates
    ppr = with_ppr.metrics["ppr"]
    assert ppr["seeded_by"] == "similarity" and ppr["seeds"] >= 1
    assert ppr["added"] == len(ids) - len(base_ids) <= 5
    assert "ppr" in with_ppr.metrics["timing"]


async def test_ppr_uses_decision_model_seeds(real_kg) -> None:
    _, relation, _ = await _single_seed_engine(real_kg, None)
    engine, _, _ = await _single_seed_engine(real_kg, KeywordDecisionModel([relation.description]))
    selected = await engine.search("query", LocalParams(
        selector="pointwise", select_keep_seed_context=True, ppr_chunks=2,
    ))
    assert selected.metrics["ppr"]["seeded_by"] == "decision_model"


async def test_query_records_generation_time(real_kg) -> None:
    engine, _, _ = await _single_seed_engine(real_kg, None)
    engine.llm = SimpleNamespace(batch_chat_completion=AsyncMock(return_value=["answer"]))
    [response] = await engine.batch_query(["query"], LocalParams())
    assert response.response == "answer"
    assert {"retrieve", "generate"} <= set(response.retrieval.metrics["timing"])


async def test_selector_without_decision_model_raises(real_kg, kg_fixture_ids) -> None:
    engine = await _engine_with_seeds(real_kg, kg_fixture_ids, None)
    with pytest.raises(ValueError, match="decision_model"):
        await engine.search("query", LocalParams(selector="pointwise"))
