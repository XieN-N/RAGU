"""
Graph search in which a decision model decides what enters the context.

:class:`DecisionSearchEngine` starts from the same seed neighbourhood as
:class:`~ragu.search_engine.local_search.LocalSearchEngine`, then replaces it
with what a single-token decision model accepts, optionally walking further hops
from the relations it kept. The decision model is mandatory: without one this
engine has no way to choose a context, so use ``LocalSearchEngine`` instead.
"""
import asyncio
import re
import time

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from textwrap import dedent
from typing import Any, Literal, get_args
from typing_extensions import override

import numpy as np
from jinja2 import Template

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.decision.decision_model import BaseDecisionModel
from ragu.decision.types import DecisionOutput

from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Relation
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import Point

from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder

from ragu.search_engine.base_engine import (
    BaseEngine,
    EngineParams,
    SearchEngineResponse,
    SearchEngineRetrieve,
    SearchEngineStreamEvent,
)
from ragu.search_engine.local_search import (
    LocalParams,
    LocalSearchEngine,
    LocalSearchResult,
    LocalSearchRetrieve,
)

Seeds = Literal["entities", "entities+facts", "entities+facts+names"]
Policy = Literal["threshold", "one_round", "best_first"]
State = Literal["facts", "chain"]
NextQuery = Literal["blob", "subquestion"]
Stop = Literal["hops", "budget", "sufficient"]
ContextOrder = Literal["sections", "ranked", "chunks_first", "flat"]
TieBreak = Literal["none", "cosine"]

#: Options with a fixed set of values, checked when parameters are constructed.
_STRATEGIES: dict[str, Any] = {
    "seeds": Seeds,
    "policy": Policy,
    "state": State,
    "next_query": NextQuery,
    "stop": Stop,
    "context_order": ContextOrder,
    "tie_break": TieBreak,
}
#: Options only ``policy="best_first"`` can honour, with their inert values.
_BEST_FIRST_ONLY = {
    "state": "facts",
    "next_query": "blob",
    "stop": "hops",
    "max_context_tokens": None,
    "max_decisions": None,
    "min_probability": None,
    "tie_break": "none",
}
#: Default budgets of ``policy="best_first"``.
DEFAULT_MAX_DECISIONS = 150
DEFAULT_MAX_CONTEXT_TOKENS = 3000
#: ``stop="sufficient"`` ends the walk once the model is this sure of the kept facts.
SUFFICIENT_PROBABILITY = 0.7
#: Longest entity names matched in a query per ``seeds="...+names"`` call.
MAX_NAME_SEEDS = 5


@dataclass
class DecisionParams(EngineParams):
    """
    Parameters for :class:`DecisionSearchEngine`.

    :param top_k: Maximum number of entities to retrieve. (Retrieval-time.)
    :param rerank_top_k: After reranking the retrieved entities, keep only this
        many most-relevant entities before deriving relations, summaries and
        chunks. ``None`` keeps all entities. (Retrieval-time.)
    :param use_summary: Whether community summaries are included in the generated
        context. (Generation-time; ignored by :meth:`DecisionSearchEngine.batch_search`.)
    :param use_chunks: Whether source chunks are included in the generated
        context. (Generation-time; ignored by :meth:`DecisionSearchEngine.batch_search`.)
    :param seeds: Where the walk starts. ``"entities"``: entities similar to the
        query. ``"entities+facts"`` adds the relations most similar to the query
        (``seed_facts`` of them), their endpoints and source chunks.
        ``"entities+facts+names"`` also adds entities named in the query, with their
        relations and source chunks.
    :param policy: How the walk uses the decision model's answers. ``"threshold"``
        keeps candidates with ``P(Y) >= select_threshold`` and expands the relations
        among them. ``"one_round"`` pools the two-hop neighbourhood by cosine similarity
        and judges it once, without state. ``"best_first"`` always expands the
        ``select_beam`` most probable unexpanded relations, whatever their ``P(Y)``,
        until a budget runs out, and fills the context with the most probable items.
    :param state: What the decision model is shown as the facts gathered so far.
        ``"facts"``: the kept facts. ``"chain"``: the relations leading to the
        candidate (``best_first`` only).
    :param next_query: What candidates of further hops are ranked against. ``"blob"``:
        the query plus the state. ``"subquestion"``: a question the LLM writes for the
        next step, also searched over the whole index (``best_first`` only).
    :param stop: When the walk ends. ``"hops"``: after ``select_hops`` hops or on the
        first hop with no passing relation. ``"budget"``: when ``max_decisions`` is
        spent or the queue is empty. ``"sufficient"``: also when the decision model says
        the kept facts answer the query.
    :param context_order: How the kept items are arranged in the context.
        ``"sections"``: entities, relations, chunks in the order they were found.
        ``"ranked"``: the same sections, each sorted by ``P(Y)``. ``"chunks_first"``:
        chunks, relations, entities, each sorted by ``P(Y)``. ``"flat"``: one list of
        all items sorted by ``P(Y)``.
    :param max_context_tokens: Context budget of ``best_first``; default
        :data:`DEFAULT_MAX_CONTEXT_TOKENS`.
    :param max_decisions: Budget of decision-model requests per query of ``best_first``;
        default :data:`DEFAULT_MAX_DECISIONS`.
    :param min_probability: Lowest ``P(Y)`` of an item ``best_first`` puts in the
        context; default ``select_threshold``.
    :param seed_facts: Relations retrieved by ``seeds="entities+facts"``.
    :param select_beam: Relations ``best_first`` expands per round.
    :param select_max_depth: Longest relation chain ``best_first`` follows.
    :param branch_pool_relations: Relations judged per expanded relation of ``best_first``.
    :param branch_pool_chunks: Chunks judged per expanded relation of ``best_first``.
    :param tie_break: ``best_first`` only: how items with nearly equal ``P(Y)`` are
        ordered, both in the expansion queue and when the context is filled.
        ``"none"``: in the order they were found. ``"cosine"``: by
        ``P(Y) + tie_weight * cosine`` to the query, so that among the many items with
        ``P(Y)`` near 0 or near 1 the ones closer to the query come first.
    :param tie_weight: Weight of the cosine in ``tie_break="cosine"``; small, so that
        it only reorders items whose ``P(Y)`` differ by less than about its value.
    :param log_decisions: ``best_first`` only: put every judged item with its text,
        ``P(Y)`` and chain length into ``metrics["decisions"]``, for diagnostics.
    :param select_pool_entities: Entities judged per hop; on the seed hop in
        retrieval order, on further hops by cosine similarity.
    :param select_pool_relations: Relations judged per hop, by cosine similarity.
    :param select_pool_chunks: Chunks judged per hop, by cosine similarity.
    :param select_threshold: Minimum ``P(Y)`` for a candidate to be kept.
    :param select_floor: On the seed hop, at least this many candidates per section
        are kept (best pool order first) even if fewer pass the threshold.
    :param select_hops: ``1`` judges the seed neighbourhood. With ``n > 1`` the
        selection walks ``n - 1`` more hops: the unvisited endpoints of relations
        that passed on the previous hop become the frontier; their entities,
        relations and chunks are pooled by cosine similarity to the query plus the
        facts kept so far, and judged with those facts as ``state_context``.
    :param select_state_max_chars: Maximum length of the kept-facts text used as
        ``state_context`` and in the hop query vector.
    :param select_keep_seed_context: Keep the full seed context instead of replacing
        it with the selection; the decision model then only picks the direction of
        further hops and the facts they add, appended after the seed context.
    """
    top_k: int = 20
    rerank_top_k: int | None = None
    use_summary: bool = False
    use_chunks: bool = True
    seeds: Seeds = "entities"
    policy: Policy = "threshold"
    state: State = "facts"
    next_query: NextQuery = "blob"
    stop: Stop = "hops"
    context_order: ContextOrder = "sections"
    max_context_tokens: int | None = None
    max_decisions: int | None = None
    min_probability: float | None = None
    select_pool_entities: int = 20
    select_pool_relations: int = 30
    select_pool_chunks: int = 10
    select_threshold: float = 0.5
    select_floor: int = 3
    select_hops: int = 1
    select_state_max_chars: int = 2000
    select_keep_seed_context: bool = False
    seed_facts: int = 20
    select_beam: int = 3
    select_max_depth: int = 4
    branch_pool_relations: int = 10
    branch_pool_chunks: int = 5
    tie_break: TieBreak = "none"
    tie_weight: float = 0.05
    log_decisions: bool = False

    def __post_init__(self) -> None:
        """
        Check the strategy options.

        :raises ValueError: If an option has a value outside its allowed set, or
            needs ``policy="best_first"``.
        """
        for name, choices in _STRATEGIES.items():
            value = getattr(self, name)
            if value not in get_args(choices):
                raise ValueError(
                    f"DecisionParams.{name}={value!r} is not one of {get_args(choices)}"
                )
        if self.policy != "best_first":
            for name, inert in _BEST_FIRST_ONLY.items():
                if getattr(self, name) != inert:
                    raise ValueError(f"DecisionParams.{name} needs policy='best_first'")


@dataclass(slots=True)
class DecisionSearchResult(LocalSearchResult):
    """
    Context chosen by the decision model.

    Same sections as :class:`~ragu.search_engine.local_search.LocalSearchResult`.
    ``entities``, ``relations`` and ``chunks`` hold the selection in pool order of
    the hop that found them; ``summaries`` and ``documents_id`` stay as retrieved.
    ``probabilities`` maps the id of every judged item to its ``P(Y)``;
    ``order_scores``, when set, overrides it for ordering the context (``P(Y)`` with
    the tie-break added);
    ``extra_entities`` counts entities in front of ``entities`` that were added to
    the retrieved ones, so that they do not push those out of the pool.
    """
    probabilities: dict[str, float] = field(default_factory=dict)
    extra_entities: int = 0
    order_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionSearchRetrieve(SearchEngineRetrieve[DecisionSearchResult]):
    """
    Retrieval container returned by :class:`DecisionSearchEngine`.

    Besides ``metrics["entities"]`` (the retrieved entities before selection), the
    metrics hold ``selection`` (per-section statistics of the seed hop),
    ``selection_hops`` (of further hops, when ``select_hops > 1``),
    ``decision_usage`` (requests and prompt size) and ``timing`` (seconds per stage).
    """
    result: DecisionSearchResult
    context_order: ContextOrder = "sections"

    _CHUNKS_FIRST_TEMPLATE = Template(dedent("""
        {%- if result.chunks %}
        **Chunks**
        {%- for c in result.chunks %}
        {{ c.content }}
        {%- endfor %}
        {% endif %}
        {%- if result.summaries %}
        **Summary**
        {%- for s in result.summaries %}
        {{ s.summary }}
        {%- endfor %}
        {% endif %}
        **Relations**
        Subject, relation type, object, relation description
        {%- for r in result.relations %}
        {{ r.subject_name }}, {{ r.relation_type }}, {{ r.object_name }} - {{ r.description }}
        {%- endfor %}

        **Entities**
        Entity, entity type, entity description
        {%- for e in result.entities %}
        {{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
        {%- endfor %}
    """))

    def to_text(self) -> str:
        """
        Render the context according to ``context_order``.

        ``"sections"`` is the layout of :class:`LocalSearchRetrieve`.
        """
        result = self.result
        if self.context_order == "sections":
            return LocalSearchRetrieve._TO_TEXT_TEMPLATE.render(result=result)

        def probability(item: Any) -> float:
            if item.id in result.order_scores:
                return -result.order_scores[item.id]
            return -result.probabilities.get(item.id, 0.0)

        if self.context_order == "flat":
            lines = [(item, _format_entity(item)) for item in result.entities]
            lines += [(item, _format_relation(item)) for item in result.relations]
            lines += [(item, _format_chunk(item)) for item in result.chunks]
            lines.sort(key=lambda pair: probability(pair[0]))
            summaries = "".join(f"{s.summary}\n" for s in result.summaries)
            return "\n**Evidence, most useful first**\n" + "\n".join(text for _, text in lines) + "\n" + summaries

        ordered = DecisionSearchResult(
            entities=sorted(result.entities, key=probability),
            relations=sorted(result.relations, key=probability),
            summaries=result.summaries,
            chunks=sorted(result.chunks, key=probability),
            documents_id=result.documents_id,
            probabilities=result.probabilities,
            order_scores=result.order_scores,
        )
        template = (
            self._CHUNKS_FIRST_TEMPLATE if self.context_order == "chunks_first"
            else LocalSearchRetrieve._TO_TEXT_TEMPLATE
        )
        return template.render(result=ordered)


class DecisionSearchEngine(BaseEngine[DecisionParams, DecisionSearchRetrieve]):
    """
    Graph search whose context is chosen by a single-token decision model.

    The engine:
      1. Retrieves the seed neighbourhood the way ``LocalSearchEngine`` does.
      2. Asks the decision model about the pooled entities, relations and chunks.
      3. Walks further hops from the relations it kept.
      4. Generates the final response from what was kept.
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        decision_model: BaseDecisionModel,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
    ):
        """
        Initialize a `DecisionSearchEngine`.

        :param llm: LLM used to generate the final answer.
        :param knowledge_graph: Knowledge graph used for retrieval.
        :param embedder: Dense embedder used for retrieval queries.
        :param decision_model: Decision model that judges every candidate. It may
            wrap ``llm`` or a separate model.
        :param sparse_embedder: Optional sparse embedder used for hybrid retrieval queries.
        :param reranker: Optional reranker used to reorder retrieved context sections.
        :param language: Default output language (fed into prompt template).
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        :raises ValueError: If ``decision_model`` is ``None``.
        """
        if decision_model is None:
            raise ValueError("DecisionSearchEngine needs a decision_model")
        super().__init__(
            llm,
            prompts=["local_search", "decision_next_query"],
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
        )
        self.knowledge_graph = knowledge_graph
        self.decision_model = decision_model
        self.language = language if language else Settings.language
        # The seed neighbourhood is exactly LocalSearch's, so it is obtained from it.
        self._seed_search = LocalSearchEngine(
            llm,
            knowledge_graph,
            embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            language=language,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
        )
        self.retriever = self._seed_search.retriever
        self._entities_by_name: dict[str, Entity] | None = None
        self._names_lock = asyncio.Lock()

    @override
    async def batch_search(
        self,
        queries: list[str],
        params: DecisionParams | None = None,
    ) -> list[DecisionSearchRetrieve]:
        """
        Gather the graph context of each query and keep what the decision model accepts.

        :param queries: Input query strings.
        :param params: Retrieval parameters. When ``None``, defaults to :class:`DecisionParams`.
        :return: One :class:`DecisionSearchRetrieve` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or DecisionParams()
        seeds = await self._seed_search.batch_search(
            queries, LocalParams(top_k=params.top_k, rerank_top_k=params.rerank_top_k),
        )
        retrieved = [
            DecisionSearchRetrieve(
                query=seed.query,
                result=DecisionSearchResult(
                    entities=seed.result.entities,
                    relations=seed.result.relations,
                    summaries=seed.result.summaries,
                    chunks=seed.result.chunks,
                    documents_id=seed.result.documents_id,
                ),
                metrics=seed.metrics,
                context_order=params.context_order,
            )
            for seed in seeds
        ]

        started = time.perf_counter()
        if params.seeds != "entities":
            await asyncio.gather(*[self._extend_seeds(query, retrieve, params) for query, retrieve in zip(queries, retrieved)])
        query_points = await self.retriever.build_query_vectors(queries)
        select = {
            "threshold": self._select_pointwise,
            "one_round": self._select_one_round,
            "best_first": self._select_best_first,
        }[params.policy]
        await asyncio.gather(*[
            select(query, point, retrieve, params)
            for query, point, retrieve in zip(queries, query_points, retrieved)
        ])
        select_seconds = time.perf_counter() - started
        for retrieve in retrieved:
            retrieve.metrics["timing"]["select"] = select_seconds
        return retrieved

    async def _selection_pools(
        self,
        query_point: Point,
        result: DecisionSearchResult,
        params: DecisionParams,
    ) -> dict[str, list[Any]]:
        """
        Candidate pools for selection: entities in retrieval order, relations and
        chunks ordered by cosine similarity to the query, each cut to its pool size.
        """
        index = self.knowledge_graph.index
        relations, chunks = await asyncio.gather(
            _order_by_query_similarity(query_point, result.relations, index.edges_vector_db),
            _order_by_query_similarity(query_point, result.chunks, index.chunks_vector_db),
        )
        return {
            "entities": result.entities[:params.select_pool_entities + result.extra_entities],
            "relations": relations[:params.select_pool_relations],
            "chunks": chunks[:params.select_pool_chunks],
        }

    async def _extend_seeds(
        self,
        query: str,
        retrieve: DecisionSearchRetrieve,
        params: DecisionParams,
    ) -> None:
        """
        Add seeds besides the entities similar to the query, per ``params.seeds``.

        :param query: Input query string.
        :param retrieve: Seed neighbourhood, extended in place.
        :param params: Selection parameters.
        """
        result = retrieve.result
        index = self.knowledge_graph.index
        [(fact_relations, _)] = await self.retriever.query_relations([query], top_k=params.seed_facts)
        relations: list[Relation] = list(fact_relations)
        entity_ids = [x for r in fact_relations for x in (r.subject_id, r.object_id)]
        chunk_ids = [c for r in fact_relations for c in r.source_chunk_id]
        named: list[Entity] = []
        if params.seeds == "entities+facts+names":
            named = await self._entities_named_in(query)
            edge_lists = await index.graph_backend.get_all_edges_for_nodes([e.id for e in named]) if named else []
            relations += [edge for edges in edge_lists for edge in edges if edge is not None]
            chunk_ids += [c for e in named for c in e.source_chunk_id]

        known = {e.id for e in result.entities}
        new_ids = list(dict.fromkeys(e for e in entity_ids if e not in known and e not in {n.id for n in named}))
        new_entities = [e for e in named if e.id not in known]
        new_entities += [e for e in await index.get_nodes(new_ids) if e is not None]
        known_relations = {r.id for r in result.relations}
        new_relations = list({r.id: r for r in relations if r.id not in known_relations}.values())
        payloads = await index.chunks_kv_storage.get_by_ids(list(dict.fromkeys(chunk_ids))) if chunk_ids else []
        known_chunks = {c.id for c in result.chunks}
        new_chunks = list({
            chunk.id: chunk for chunk in (Chunk(**payload) for payload in payloads if payload is not None)
            if chunk.id not in known_chunks
        }.values())
        result.entities = new_entities + result.entities
        result.extra_entities = len(new_entities)
        result.relations = result.relations + new_relations
        result.chunks = result.chunks + new_chunks
        retrieve.metrics["seed_extension"] = {
            "entities": len(new_entities), "named": len(named),
            "relations": len(new_relations), "chunks": len(new_chunks),
        }

    async def _entities_named_in(self, query: str) -> list[Entity]:
        """
        Entities whose name occurs in the query as whole words, longest names first.

        Names shorter than four characters or inside a longer match are skipped.

        :param query: Input query string.
        :return: At most :data:`MAX_NAME_SEEDS` entities.
        """
        async with self._names_lock:
            if self._entities_by_name is None:
                entities = await self.knowledge_graph.index.graph_backend.get_all_nodes()
                self._entities_by_name = {
                    e.entity_name.strip().lower(): e
                    for e in entities if e is not None and len(e.entity_name.strip()) >= 4
                }
        lowered = query.lower()
        matches: list[str] = []
        for name in sorted((n for n in self._entities_by_name if n in lowered), key=len, reverse=True):
            if any(name in longer for longer in matches):
                continue
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
                matches.append(name)
        return [self._entities_by_name[name] for name in matches[:MAX_NAME_SEEDS]]

    async def _select_one_round(
        self,
        query: str,
        query_point: Point,
        retrieve: DecisionSearchRetrieve,
        params: DecisionParams,
    ) -> None:
        """
        Pool the two-hop neighbourhood by similarity and judge it once, without state.

        The neighbours are those of the relations most similar to the query, so that
        the neighbourhood of a hub does not dominate.

        :param query: Input query string.
        :param query_point: Query vector built by the retriever.
        :param retrieve: Seed neighbourhood, extended and then replaced in place.
        :param params: Selection parameters.
        """
        result = retrieve.result
        index = self.knowledge_graph.index
        pools = await self._selection_pools(query_point, result, params)
        known = {e.id for e in result.entities}
        neighbour_ids = sorted({x for r in pools["relations"] for x in (r.subject_id, r.object_id)} - known)
        neighbours = [e for e in await index.get_nodes(neighbour_ids) if e is not None]
        if neighbours:
            edge_lists = await index.graph_backend.get_all_edges_for_nodes([e.id for e in neighbours])
            known_relations = {r.id for r in result.relations}
            relations = {
                edge.id: edge for edges in edge_lists for edge in edges
                if edge is not None and edge.id not in known_relations
            }
            chunk_ids = sorted({c for e in neighbours for c in e.source_chunk_id})
            payloads = await index.chunks_kv_storage.get_by_ids(chunk_ids) if chunk_ids else []
            known_chunks = {c.id for c in result.chunks}
            chunks = {
                chunk.id: chunk for chunk in (Chunk(**payload) for payload in payloads if payload is not None)
                if chunk.id not in known_chunks
            }
            neighbours = await _order_by_query_similarity(query_point, neighbours, index.nodes_vector_db)
            result.entities = result.entities + neighbours
            result.relations = result.relations + list(relations.values())
            result.chunks = result.chunks + list(chunks.values())
        await self._select_pointwise(query, query_point, retrieve, replace(params, select_hops=1))

    async def _select_best_first(
        self,
        query: str,
        query_point: Point,
        retrieve: DecisionSearchRetrieve,
        params: DecisionParams,
    ) -> None:
        """
        Expand the most probable relations until a budget runs out.

        Unlike ``threshold``, a relation does not need to pass anything to be expanded,
        so a strict model does not end the walk. The context is filled afterwards with
        the most probable judged items, within ``max_context_tokens``.

        :param query: Input query string.
        :param query_point: Query vector built by the retriever.
        :param retrieve: Seed neighbourhood, replaced in place.
        :param params: Selection parameters.
        """
        result = retrieve.result
        index = self.knowledge_graph.index
        max_decisions = params.max_decisions if params.max_decisions is not None else DEFAULT_MAX_DECISIONS
        pools = await self._selection_pools(query_point, result, params)
        expanded = {e.id for e in result.entities}
        outputs: list[DecisionOutput[Any]] = list(await self._score_sections(query, pools, params.select_threshold))
        judged = {item.id for section in pools.values() for item in section}
        scored: dict[str, list[Any]] = {section: [] for section in _SECTIONS}
        queue: list[tuple[float, Relation, tuple[Relation, ...]]] = []
        decisions: list[list[Any]] = []
        similarity: dict[str, float] = {}
        tie_weight = params.tie_weight if params.tie_break == "cosine" else 0.0

        def absorb(section: str, output: DecisionOutput[Any], path: tuple[Relation, ...]) -> None:
            _record_probabilities(result, output)
            for candidate in output.scored:
                if candidate.degraded:
                    continue
                scored[section].append(candidate)
                if params.log_decisions:
                    decisions.append([section, candidate.probability, len(path), _FORMATTERS[section](candidate.item)])
                if section == "relations":
                    queue.append((candidate.probability, candidate.item, (*path, candidate.item)))

        for section, output in zip(_SECTIONS, outputs):
            absorb(section, output, ())

        rounds = 0
        stopped_by = "queue"
        while queue:
            spent = sum(o.n_calls for o in outputs)
            if spent >= max_decisions:
                stopped_by = "budget"
                break
            if tie_weight:
                await self._add_similarities(query_point, "relations", [entry[1] for entry in queue], similarity)
            queue.sort(key=lambda entry: -(entry[0] + tie_weight * similarity.get(entry[1].id, 0.0)))
            batch, queue = queue[:params.select_beam], queue[params.select_beam:]
            state = _top_facts_text(scored, params.select_state_max_chars)
            if params.stop == "sufficient" and rounds and await self._facts_suffice(query, state):
                stopped_by = "sufficient"
                break
            subquestion = ""
            if params.next_query == "subquestion":
                subquestion = await self._next_subquestion(query, state)
            rounds += 1

            # Which branch gets which node or candidate must not depend on how long
            # storage calls take: nodes are claimed before the branches start and
            # candidates after they finish, in queue order.
            found_by_subquestion: list[Relation] = []
            subquestion_point: Point | None = None
            if subquestion and any(len(path) <= params.select_max_depth for _, _, path in batch):
                [(found_by_subquestion, _)] = await self.retriever.query_relations(
                    [subquestion], top_k=params.branch_pool_relations,
                )
                [subquestion_point] = await self.retriever.build_query_vectors([subquestion])
            frontier_of: list[list[str]] = []
            for _, relation, path in batch:
                claimed: list[str] = []
                if len(path) <= params.select_max_depth:
                    claimed = sorted({relation.subject_id, relation.object_id} - expanded)
                    expanded.update(claimed)
                frontier_of.append(claimed)

            async def collect(
                entry: tuple[float, Relation, tuple[Relation, ...]],
                frontier_ids: list[str],
            ) -> tuple[tuple[Relation, ...], str, list[Any], list[Any], list[Any]] | None:
                _, _, path = entry
                if len(path) > params.select_max_depth:
                    return None
                frontier = [e for e in await index.get_nodes(frontier_ids) if e is not None] if frontier_ids else []
                relations: dict[str, Relation] = {}
                chunk_ids: set[str] = set()
                if frontier:
                    edge_lists = await index.graph_backend.get_all_edges_for_nodes([e.id for e in frontier])
                    relations = {
                        edge.id: edge for edges in edge_lists for edge in edges
                        if edge is not None and edge.id not in judged
                    }
                    chunk_ids = {c for e in frontier for c in e.source_chunk_id}
                branch_state = _chain_text(path) if params.state == "chain" else state
                relations.update({r.id: r for r in found_by_subquestion if r.id not in judged})
                chunk_ids |= {c for r in found_by_subquestion for c in r.source_chunk_id}
                payloads = await index.chunks_kv_storage.get_by_ids(sorted(chunk_ids)) if chunk_ids else []
                chunks = [
                    chunk for chunk in (Chunk(**payload) for payload in payloads if payload is not None)
                    if chunk.id not in judged
                ]
                entities = [e for e in frontier if e.id not in judged]
                if not (entities or relations or chunks):
                    return None
                point = subquestion_point
                if point is None:
                    [point] = await self.retriever.build_query_vectors([f"{query}\n{branch_state}"])
                ordered_relations, ordered_chunks = await asyncio.gather(
                    _order_by_query_similarity(point, list(relations.values()), index.edges_vector_db),
                    _order_by_query_similarity(point, chunks, index.chunks_vector_db),
                )
                return path, branch_state, entities, ordered_relations, ordered_chunks

            async def score(
                path: tuple[Relation, ...],
                branch_state: str,
                branch_pools: dict[str, list[Any]],
            ) -> list[tuple[str, DecisionOutput[Any], tuple[Relation, ...]]]:
                branch_outputs = await self._score_sections(
                    query, branch_pools, params.select_threshold, state_context=branch_state,
                )
                return [(section, output, path) for section, output in zip(_SECTIONS, branch_outputs)]

            allowance_per_branch = max(1, (max_decisions - spent) // max(1, len(batch)))
            collected = await asyncio.gather(*[collect(entry, ids) for entry, ids in zip(batch, frontier_of)])
            scoring = []
            for branch in collected:
                if branch is None:
                    continue
                path, branch_state, entities, ordered_relations, ordered_chunks = branch
                allowance = allowance_per_branch
                entity_pool = [e for e in entities if e.id not in judged][:allowance]
                allowance -= len(entity_pool)
                relation_pool = [r for r in ordered_relations if r.id not in judged][
                    :min(params.branch_pool_relations, allowance)
                ]
                allowance -= len(relation_pool)
                chunk_pool = [c for c in ordered_chunks if c.id not in judged][
                    :min(params.branch_pool_chunks, allowance)
                ]
                for item in (*entity_pool, *relation_pool, *chunk_pool):
                    judged.add(item.id)
                if entity_pool or relation_pool or chunk_pool:
                    scoring.append(score(path, branch_state, {
                        "entities": entity_pool, "relations": relation_pool, "chunks": chunk_pool,
                    }))
            for branch in await asyncio.gather(*scoring):
                for section, output, path in branch:
                    outputs.append(output)
                    absorb(section, output, path)

        if tie_weight:
            for section, candidates in scored.items():
                await self._add_similarities(query_point, section, [c.item for c in candidates], similarity)
        self._fill_context(retrieve, scored, params, similarity if tie_weight else None)
        if params.log_decisions:
            retrieve.metrics["decisions"] = decisions
        retrieve.metrics["selection_walk"] = {
            "rounds": rounds,
            "stopped_by": stopped_by,
            "expanded_entities": len(expanded),
            **{f"judged_{name}": len(items) for name, items in scored.items()},
            **{f"kept_{name}": len(getattr(result, name)) for name in scored},
        }
        retrieve.metrics["decision_usage"] = {
            "calls": sum(o.n_calls for o in outputs),
            "prompt_chars": sum(o.prompt_chars for o in outputs),
        }

    async def _score_sections(
        self,
        query: str,
        pools: dict[str, list[Any]],
        threshold: float,
        state_context: str = "",
    ) -> list[DecisionOutput[Any]]:
        """
        Judge the entity, relation and chunk pools concurrently.

        :param query: Input query string.
        :param pools: Candidates per section.
        :param threshold: Minimum ``P(Y)`` for a candidate to pass.
        :param state_context: Facts gathered so far, shown to the decision model.
        :return: One output per section, in the order of :data:`_SECTIONS`.
        """
        return await asyncio.gather(*[
            self.decision_model.a_score_pointwise(
                query, pools[section], _FORMATTERS[section], state_context=state_context, threshold=threshold,
            )
            for section in _SECTIONS
        ])

    async def _add_similarities(
        self,
        query_point: Point,
        section: str,
        items: list[Any],
        similarity: dict[str, float],
    ) -> None:
        """
        Add the cosine to the query of items not yet in ``similarity``.

        Stored vectors are used where the index has them; the rest (e.g. relations of
        an index built without relation vectors) are embedded from their text.

        :param query_point: Query vector.
        :param section: ``"entities"``, ``"relations"`` or ``"chunks"``.
        :param items: Items with an ``id``.
        :param similarity: Map from item id to cosine, updated in place.
        """
        todo = list({item.id: item for item in items if item.id not in similarity}.values())
        if not todo or query_point.dense_embedding is None:
            return
        index = self.knowledge_graph.index
        vector_db = {
            "entities": index.nodes_vector_db,
            "relations": index.edges_vector_db,
            "chunks": index.chunks_vector_db,
        }[section]
        query = np.asarray(query_point.dense_embedding, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        points = await vector_db.get_points_by_ids([item.id for item in todo])
        vectors = {
            item.id: point.dense_embedding for item, point in zip(todo, points)
            if point is not None and point.dense_embedding is not None
        }
        missing = [item for item in todo if item.id not in vectors]
        if missing:
            embedded = await self.retriever.build_query_vectors([_FORMATTERS[section](item) for item in missing])
            vectors.update({item.id: point.dense_embedding for item, point in zip(missing, embedded)})
        for item_id, vector in vectors.items():
            vector = np.asarray(vector, dtype=np.float32)
            similarity[item_id] = float(vector @ query / (np.linalg.norm(vector) or 1.0))

    def _fill_context(
        self,
        retrieve: DecisionSearchRetrieve,
        scored: dict[str, list[Any]],
        params: DecisionParams,
        similarity: dict[str, float] | None = None,
    ) -> None:
        """
        Replace the context with the most probable judged items within the token budget.

        :param retrieve: Retrieval container, changed in place.
        :param scored: Judged candidates per section.
        :param params: Selection parameters.
        :param similarity: Cosine of items to the query for ``tie_break="cosine"``;
            ``None`` orders by ``P(Y)`` alone.
        """
        result = retrieve.result
        min_probability = params.min_probability if params.min_probability is not None else params.select_threshold
        budget = params.max_context_tokens if params.max_context_tokens is not None else DEFAULT_MAX_CONTEXT_TOKENS
        weight = params.tie_weight if similarity is not None else 0.0
        cosine = similarity or {}

        def order_score(probability: float, item: Any) -> float:
            return probability + weight * cosine.get(item.id, 0.0)

        ranked = sorted(
            ((c.probability, name, c.item) for name, items in scored.items() for c in items),
            key=lambda entry: -order_score(entry[0], entry[2]),
        )
        kept: dict[str, list[Any]] = {"entities": [], "relations": [], "chunks": []}
        tokens = 0
        for probability, name, item in ranked:
            if probability < min_probability:
                continue
            size = len(self.truncation.encoder.encode(_FORMATTERS[name](item)))
            if tokens + size > budget:
                continue
            tokens += size
            kept[name].append(item)
            if weight:
                result.order_scores[item.id] = order_score(probability, item)
        result.entities, result.relations, result.chunks = kept["entities"], kept["relations"], kept["chunks"]

    async def _facts_suffice(self, query: str, facts: str) -> bool:
        """Ask the decision model whether the kept facts answer the query."""
        decision = await self.decision_model.a_decide(
            f"Do these facts contain the answer to the query, or every step needed to derive it?\nQuery: {query}",
            ["Yes", "No"],
            context=facts,
        )
        return not decision.degraded and decision.distribution.get("Yes", 0.0) >= SUFFICIENT_PROBABILITY

    async def _next_subquestion(self, query: str, facts: str) -> str:
        """Have the LLM write the next factual question to look up."""
        instruction: RAGUInstruction = self.get_prompt("decision_next_query")
        conversation = render(instruction.messages, query=query, facts=facts)[0].to_openai()
        [text] = await self.llm.batch_chat_completion(
            [conversation], output_schema=instruction.pydantic_model, desc="DecisionSearch next query",
        )
        return (text or "").strip() or query

    async def _select_pointwise(
        self,
        query: str,
        query_point: Point,
        retrieve: DecisionSearchRetrieve,
        params: DecisionParams,
    ) -> None:
        """
        Keep only the entities, relations and chunks the decision model judges useful.

        Relations and chunks are first ordered by cosine similarity to the query
        so that the judged pools hold the most similar candidates. The selection
        replaces the sections of ``retrieve.result`` in place (unless
        ``params.select_keep_seed_context``) and is reported in
        ``retrieve.metrics["selection"]``; request volume goes to
        ``retrieve.metrics["decision_usage"]``.

        :param query: Input query string.
        :param query_point: Query vector built by the retriever.
        :param retrieve: Assembled seed context of the query.
        :param params: Selection parameters.
        """
        result = retrieve.result
        seen_ids = (
            {e.id for e in result.entities}
            | {r.id for r in result.relations}
            | {c.id for c in result.chunks}
        )
        pools = await self._selection_pools(query_point, result, params)
        entity_out, relation_out, chunk_out = await self._score_sections(query, pools, params.select_threshold)
        for output in (entity_out, relation_out, chunk_out):
            _record_probabilities(result, output)
        # What the decision model accepted: the state for further hops when the
        # seed context is kept whole.
        kept = DecisionSearchResult(
            entities=list(entity_out.selected),
            relations=list(relation_out.selected),
            chunks=list(chunk_out.selected),
        )
        if not params.select_keep_seed_context:
            result.entities = _keep_with_floor(entity_out, pools["entities"], params.select_floor)
            result.relations = _keep_with_floor(relation_out, pools["relations"], params.select_floor)
            result.chunks = _keep_with_floor(chunk_out, pools["chunks"], params.select_floor)
        retrieve.metrics["selection"] = {
            name: {
                "pool": len(pools[name]),
                "kept": len(section),
                "passed": len(output.selected),
                "n_calls": output.n_calls,
                "degraded": output.degraded_count,
                "probabilities": [round(c.probability, 4) for c in output.scored],
            }
            for name, output, section in (
                ("entities", entity_out, result.entities),
                ("relations", relation_out, result.relations),
                ("chunks", chunk_out, result.chunks),
            )
        }
        outputs = [entity_out, relation_out, chunk_out]

        if params.select_hops > 1:
            started = time.perf_counter()
            # The state is the selection, floor items included. With the seed context
            # kept whole, only what the model accepted counts as state.
            state_source = kept if params.select_keep_seed_context else result
            outputs += await self._select_further_hops(
                query, retrieve, params, list(relation_out.selected), seen_ids, state_source,
            )
            retrieve.metrics["timing"]["hops"] = time.perf_counter() - started

        retrieve.metrics["decision_usage"] = {
            "calls": sum(o.n_calls for o in outputs),
            "prompt_chars": sum(o.prompt_chars for o in outputs),
        }

    async def _select_further_hops(
        self,
        query: str,
        retrieve: DecisionSearchRetrieve,
        params: DecisionParams,
        passed_relations: list[Relation],
        seen_ids: set[str],
        state_source: DecisionSearchResult,
    ) -> list[DecisionOutput[Any]]:
        """
        Walk ``params.select_hops - 1`` hops beyond the seeds, guided by the decision model.

        The frontier of each hop is the unvisited endpoints of the relations that
        passed on the previous hop. Its entities, incident relations and source
        chunks (all unseen so far) are pooled by cosine similarity to the query
        plus the kept facts and judged with those facts as ``state_context``.
        Passing items are appended to ``retrieve.result`` (and to ``state_source``
        when it is a separate view); per-hop statistics go to
        ``retrieve.metrics["selection_hops"]``.

        :param query: Input query string.
        :param retrieve: Context after the seed-hop selection, extended in place.
        :param params: Selection parameters.
        :param passed_relations: Seed-hop relations that passed the threshold.
        :param seen_ids: Ids of every entity, relation and chunk already assembled.
        :param state_source: Facts rendered as the decision-model state.
        :return: Decision outputs of every judged section, for usage accounting.
        """
        result = retrieve.result
        index = self.knowledge_graph.index
        visited = set(seen_ids)
        hop_stats: list[dict[str, Any]] = []
        outputs: list[DecisionOutput[Any]] = []

        for hop in range(1, params.select_hops):
            frontier_ids = sorted(
                {x for r in passed_relations for x in (r.subject_id, r.object_id)} - visited
            )
            if not frontier_ids:
                break
            visited.update(frontier_ids)
            frontier = [e for e in await index.get_nodes(frontier_ids) if e is not None]

            edge_lists = await index.graph_backend.get_all_edges_for_nodes([e.id for e in frontier])
            relations: dict[str, Relation] = {}
            for edges in edge_lists:
                for edge in edges:
                    if edge is not None and edge.id not in seen_ids:
                        relations[edge.id] = edge
            chunk_ids = sorted(
                {c_id for e in frontier for c_id in e.source_chunk_id} - seen_ids
            )
            payloads = await index.chunks_kv_storage.get_by_ids(chunk_ids) if chunk_ids else []
            # Storage keys and content-derived Chunk ids can differ; dedupe on the latter.
            chunks = [
                chunk for chunk in (Chunk(**payload) for payload in payloads if payload is not None)
                if chunk.id not in seen_ids
            ]
            entities = [e for e in frontier if e.id not in seen_ids]
            seen_ids.update(relations)
            seen_ids.update(c.id for c in chunks)
            seen_ids.update(e.id for e in entities)

            state = _state_text(state_source, params.select_state_max_chars)
            [state_point] = await self.retriever.build_query_vectors([f"{query}\n{state}"])
            ordered_entities, ordered_relations, ordered_chunks = await asyncio.gather(
                _order_by_query_similarity(state_point, entities, index.nodes_vector_db),
                _order_by_query_similarity(state_point, list(relations.values()), index.edges_vector_db),
                _order_by_query_similarity(state_point, chunks, index.chunks_vector_db),
            )
            pools: dict[str, list[Any]] = {
                "entities": ordered_entities[:params.select_pool_entities],
                "relations": ordered_relations[:params.select_pool_relations],
                "chunks": ordered_chunks[:params.select_pool_chunks],
            }
            entity_out, relation_out, chunk_out = await self._score_sections(
                query, pools, params.select_threshold, state_context=state,
            )
            outputs += [entity_out, relation_out, chunk_out]
            for output in (entity_out, relation_out, chunk_out):
                _record_probabilities(result, output)
            for section, output in (
                ("entities", entity_out), ("relations", relation_out), ("chunks", chunk_out),
            ):
                getattr(result, section).extend(output.selected)
                if state_source is not result:
                    getattr(state_source, section).extend(output.selected)
            hop_stats.append({
                "hop": hop,
                "frontier": len(frontier),
                **{
                    name: {
                        "candidates": total,
                        "pool": len(pools[name]),
                        "passed": len(output.selected),
                        "n_calls": output.n_calls,
                        "degraded": output.degraded_count,
                    }
                    for name, output, total in (
                        ("entities", entity_out, len(entities)),
                        ("relations", relation_out, len(relations)),
                        ("chunks", chunk_out, len(chunks)),
                    )
                },
            })
            passed_relations = list(relation_out.selected)
            if not passed_relations:
                break
        retrieve.metrics["selection_hops"] = hop_stats
        return outputs

    def _render_answer_messages(
        self,
        query: str,
        context: DecisionSearchRetrieve,
        use_summary: bool,
        use_chunks: bool,
    ) -> ChatMessages:
        """
        Build the final-answer conversation from the chosen context.

        Prunes summaries/chunks according to the flags, truncates the rendered
        context to the engine token limit, and renders the ``local_search`` prompt.

        :param query: User query in natural language.
        :param context: Retrieval container produced by :meth:`search`.
        :param use_summary: Whether community summaries are kept in the context.
        :param use_chunks: Whether source chunks are kept in the context.
        :return: Rendered chat messages ready for ``chat_completion``.
        """
        if not use_summary:
            context.result.summaries = []
        if not use_chunks:
            context.result.chunks = []

        instruction: RAGUInstruction = self.get_prompt("local_search")
        return render(
            instruction.messages,
            query=query,
            context=self.truncation(str(context)),
            language=self.language,
        )[0]

    @override
    async def batch_query(
        self,
        queries: list[str],
        params: DecisionParams | None = None,
    ) -> list[SearchEngineResponse]:
        """
        Answer each query from the context the decision model chose for it.

        :param queries: User queries in natural language.
        :param params: Query parameters. When ``None``, defaults to :class:`DecisionParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or DecisionParams()
        contexts = await self.batch_search(queries, params)

        conversations = [
            self._render_answer_messages(query, context, params.use_summary, params.use_chunks).to_openai()
            for query, context in zip(queries, contexts)
        ]

        instruction: RAGUInstruction = self.get_prompt("local_search")
        started = time.perf_counter()
        answers = await self.llm.batch_chat_completion(
            conversations,
            output_schema=instruction.pydantic_model,
            desc="DecisionSearch batch query",
        )
        generate_seconds = time.perf_counter() - started
        for context in contexts:
            context.metrics.setdefault("timing", {})["generate"] = generate_seconds

        return [
            SearchEngineResponse(
                query=query,
                response=answer,
                retrieval=context,
                payload={},
            )
            for query, answer, context in zip(queries, answers, contexts)
        ]

    @override
    async def stream_query(
        self,
        query: str,
        params: DecisionParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Choose the context with the decision model and stream the plain-text answer.

        :param query: User query in natural language.
        :param params: Query parameters. When ``None``, defaults to :class:`DecisionParams`.
        :returns: Async iterator of text deltas with the retrieval context.
        """
        params = params or DecisionParams()
        context = await self.search(query, params)
        conversation = self._render_answer_messages(
            query,
            context,
            params.use_summary,
            params.use_chunks,
        ).to_openai()

        async for delta in self.llm.stream_chat_completion(conversation):
            yield SearchEngineStreamEvent(
                query=query,
                retrieval=context,
                delta=delta,
                payload={},
            )


def _format_entity(entity: Entity) -> str:
    return f"{entity.entity_name} ({entity.entity_type}): {entity.description}"


def _format_relation(relation: Relation) -> str:
    return (
        f"{relation.subject_name} --[{relation.relation_type}]--> "
        f"{relation.object_name}: {relation.description}"
    )


def _format_chunk(chunk: Chunk) -> str:
    return chunk.content


_SECTIONS = ("entities", "relations", "chunks")
_FORMATTERS = {"entities": _format_entity, "relations": _format_relation, "chunks": _format_chunk}


def _state_text(result: DecisionSearchResult, max_chars: int) -> str:
    """
    Facts kept so far, as decision-model state: relations first, then entities.

    :param result: Current context.
    :param max_chars: Maximum length of the returned text.
    :return: Newline-separated facts, cut to ``max_chars``.
    """
    lines = [_format_relation(r) for r in result.relations]
    lines += [_format_entity(e) for e in result.entities]
    return "\n".join(lines)[:max_chars]


async def _order_by_query_similarity(
    query_point: Point,
    items: list[Any],
    vector_db: BaseVectorStorage,
) -> list[Any]:
    """
    Sort items by cosine similarity of their stored vectors to the query.

    Items without a stored vector keep their relative order after the scored ones.

    :param query_point: Query vector.
    :param items: Items with an ``id`` attribute matching ``vector_db`` ids.
    :param vector_db: Vector storage holding the items' embeddings.
    :return: Items, most similar first.
    """
    if not items or query_point.dense_embedding is None:
        return list(items)
    query = np.asarray(query_point.dense_embedding, dtype=np.float32)
    query = query / (np.linalg.norm(query) or 1.0)
    points = await vector_db.get_points_by_ids([item.id for item in items])
    scored: list[tuple[float, int]] = []
    missing: list[int] = []
    for i, point in enumerate(points):
        if point is None or point.dense_embedding is None:
            missing.append(i)
            continue
        vector = np.asarray(point.dense_embedding, dtype=np.float32)
        scored.append((float(vector @ query / (np.linalg.norm(vector) or 1.0)), i))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [items[i] for _, i in scored] + [items[i] for i in missing]


def _top_facts_text(scored: dict[str, list[Any]], max_chars: int) -> str:
    """
    Most probable judged relations and entities, as decision-model state.

    :param scored: Judged candidates per section.
    :param max_chars: Maximum length of the returned text.
    :return: Newline-separated facts, most probable first, cut to ``max_chars``.
    """
    entries = [(c.probability, _format_relation(c.item)) for c in scored["relations"] if c.passed_threshold]
    entries += [(c.probability, _format_entity(c.item)) for c in scored["entities"] if c.passed_threshold]
    entries.sort(key=lambda entry: -entry[0])
    return "\n".join(text for _, text in entries)[:max_chars]


def _chain_text(path: tuple[Relation, ...]) -> str:
    """The relations leading to a candidate, first hop first."""
    return "\n".join(_format_relation(relation) for relation in path)


def _record_probabilities(result: DecisionSearchResult, output: DecisionOutput[Any]) -> None:
    """Remember ``P(Y)`` of every judged item, for ordering the context."""
    for candidate in output.scored:
        result.probabilities[candidate.item.id] = candidate.probability


def _keep_with_floor(output: DecisionOutput[Any], pool: list[Any], floor: int) -> list[Any]:
    """
    Selected candidates, topped up from the pool order up to ``floor`` items.

    :param output: Decision model output over ``pool``.
    :param pool: Judged candidates in pool order.
    :param floor: Minimum number of items to return (bounded by the pool size).
    :return: Kept items, selected ones first.
    """
    kept = list(output.selected)
    if len(kept) < floor:
        chosen = {id(item) for item in kept}
        for item in pool:
            if len(kept) >= floor:
                break
            if id(item) not in chosen:
                kept.append(item)
                chosen.add(id(item))
    return kept
