# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/
import asyncio

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Literal
from typing_extensions import override

import numpy as np
from jinja2 import Template

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.decision.decision_model import BaseDecisionModel
from ragu.decision.types import DecisionOutput

from ragu.graph.graph_retrieve_backend import GraphRetriever
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity, Relation
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import EmbeddingHit, Point

from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder

from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineRetrieve,
    SearchEngineResponse,
    SearchEngineStreamEvent,
    EngineParams,
)
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_documents_id,
    _find_most_related_community_from_entities,
    _rerank_items,
    _prefetch_entity_edges,
)


@dataclass
class LocalParams(EngineParams):
    """
    Parameters for :class:`LocalSearchEngine`.

    :param top_k: Maximum number of entities to retrieve. (Retrieval-time.)
    :param rerank_top_k: After reranking the retrieved entities, keep only this
        many most-relevant entities before deriving relations, summaries and
        chunks. ``None`` keeps all entities. (Retrieval-time.)
    :param use_summary: Whether community summaries are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    :param use_chunks: Whether source chunks are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    :param selector: ``"none"`` keeps the full local context. ``"pointwise"``
        asks the engine's decision model a ``Y``/``N`` relevance question for
        each candidate entity, relation and chunk and keeps the ones that pass.
        (Retrieval-time; requires ``decision_model``.) ``"cosine"`` builds the
        same pools and keeps the ``select_top_*`` most query-similar candidates
        of each section, without a decision model (a ranking baseline).
    :param select_top_entities: Entities kept by the ``"cosine"`` selector.
    :param select_top_relations: Relations kept by the ``"cosine"`` selector.
    :param select_top_chunks: Chunks kept by the ``"cosine"`` selector.
    :param select_pool_entities: Entities judged by the selector, in retrieval
        order; the rest are dropped.
    :param select_pool_relations: Relations judged by the selector, taken in
        order of cosine similarity to the query; the rest are dropped.
    :param select_pool_chunks: Chunks judged by the selector, taken in order of
        cosine similarity to the query; the rest are dropped.
    :param select_threshold: Minimum ``P(Y)`` for a candidate to be kept.
    :param select_floor: Per section, at least this many candidates are kept
        (best pool order first) even if fewer pass the threshold.
    """
    top_k: int = 20
    rerank_top_k: int | None = None
    use_summary: bool = False
    use_chunks: bool = True
    selector: Literal["none", "pointwise", "cosine"] = "none"
    select_pool_entities: int = 20
    select_pool_relations: int = 30
    select_pool_chunks: int = 10
    select_threshold: float = 0.5
    select_floor: int = 3
    select_top_entities: int = 4
    select_top_relations: int = 4
    select_top_chunks: int = 3


@dataclass(slots=True)
class LocalSearchResult:
    """
    Retrieved graph-local context for a query.

    Entities are the seed retrieval results. Relations, summaries, and chunks
    are derived from those entities and optionally reranked. ``documents_id``
    contains unique source document IDs from the final entity set.
    """
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    summaries: list[Any] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    documents_id: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocalSearchRetrieve(SearchEngineRetrieve[LocalSearchResult]):
    """
    Retrieval container returned by :class:`LocalSearchEngine`.

    Metrics use ``metrics["entities"]`` with one entry per final entity
    containing ``id``, ``name``, zero-based ``rank``, and vector
    ``relevance_score``.
    """
    result: LocalSearchResult

    _TO_TEXT_TEMPLATE = Template(dedent("""
        **Entities**
        Entity, entity type, entity description
        {%- for e in result.entities %}
        {{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
        {%- endfor %}

        **Relations**
        Subject, relation type, object, relation description, rank
        {%- for r in result.relations %}
        {{ r.subject_name }}, {{ r.relation_type }}, {{ r.object_name }} - {{ r.description }}, {{ r.rank }}
        {%- endfor %}

        {%- if result.summaries %}
        **Summary**
        {%- for s in result.summaries %}
        {{ s.summary }}
        {%- endfor %}
        {% endif %}

        {%- if result.chunks %}
        **Chunks**
        {%- for c in result.chunks %}
        {{ c.content }}
        {%- endfor %}
        {% endif %}
    """))

    def to_text(self) -> str:
        """
        Render entities, relations, optional summaries, and optional chunks.
        """
        return self._TO_TEXT_TEMPLATE.render(result=self.result)


class LocalSearchEngine(BaseEngine[LocalParams, LocalSearchRetrieve]):
    """
    Performs local retrieval-augmented search (RAG) over a knowledge graph.

    The engine:
      1. Retrieves relevant entities for the query.
      2. Retrieves related items (relations, summary and chunks).
      3. Generates a final response

    Reference
    ---------
    Based on: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None,
        language: str | None = None,
        max_context_length: int | None = None,
        tokenizer_backend: Literal["tiktoken", "local"] | None = None,
        tokenizer_model: str | None = None,
        decision_model: BaseDecisionModel | None = None,
    ):
        """
        Initialize a `LocalSearchEngine`.

        :param llm: LLM used to generate the final answer.
        :param knowledge_graph: Knowledge graph used for entity and relation retrieval.
        :param embedder: Dense embedder used for retrieval queries.
        :param sparse_embedder: Optional sparse embedder used for hybrid retrieval queries.
        :param reranker: Optional reranker used to reorder retrieved context sections.
        :param language: Default output language (fed into prompt template).
        :param max_context_length: Maximum tokens for the assembled context fed to
            the LLM. When ``None``, falls back to ``Settings.llm_context_token_limit``.
        :param tokenizer_backend: Tokenizer backend for context truncation. When
            ``None``, falls back to ``Settings.tokenizer_llm_backend``.
        :param tokenizer_model: Tokenizer model identifier for context truncation.
            When ``None``, falls back to ``Settings.tokenizer_llm_name``.
        :param decision_model: Optional single-token decision model used when
            ``LocalParams.selector`` is not ``"none"``. It may wrap ``llm`` or a
            separate model.
        """
        _PROMPTS_NAMES = ["local_search"]
        super().__init__(
            llm,
            prompts=_PROMPTS_NAMES,
            max_context_length=max_context_length,
            tokenizer_backend=tokenizer_backend,
            tokenizer_model=tokenizer_model,
        )

        self.knowledge_graph = knowledge_graph
        self.retriever = GraphRetriever(
            knowledge_graph=knowledge_graph,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
        )
        self.reranker = reranker
        self.decision_model = decision_model
        self.language = language if language else Settings.language

    @override
    async def batch_search(
        self,
        queries: List[str],
        params: LocalParams | None = None,
    ) -> List[LocalSearchRetrieve]:
        """
        Find the entities most relevant to each query and gather the graph context around them.

        For every query, locates the knowledge-graph entities closest in meaning to it and
        collects their local neighborhood — the relations they take part in, the text chunks
        they were extracted from, the summaries of the communities they belong to, and the
        documents they originate from.

        :param queries: Input query strings.
        :param params: Retrieval parameters. When ``None``, defaults to :class:`LocalParams`.
        :return: One :class:`LocalSearchRetrieve` per query, aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or LocalParams()
        if params.selector == "pointwise" and self.decision_model is None:
            raise ValueError(
                f"LocalParams.selector={params.selector!r} needs a decision_model "
                f"passed to LocalSearchEngine"
            )
        entity_results = await self.retriever.query_entities(queries, top_k=params.top_k)

        ranked_entities = await asyncio.gather(*[
            self._rank_entities(query, entities, params.rerank_top_k)
            for query, (entities, _) in zip(queries, entity_results)
        ])

        edges_by_entity, neighbor_chunks = await _prefetch_entity_edges(
            [entity.id for entities in ranked_entities for entity in entities if entity and entity.id],
            self.knowledge_graph,
        )

        retrieved = list(await asyncio.gather(*[
            self._assemble(query, entities, entity_hits, ranked, edges_by_entity, neighbor_chunks)
            for query, (entities, entity_hits), ranked
            in zip(queries, entity_results, ranked_entities)
        ]))

        if params.selector != "none":
            select = self._select_pointwise if params.selector == "pointwise" else self._select_cosine
            query_points = await self.retriever.build_query_vectors(queries)
            await asyncio.gather(*[
                select(query, point, retrieve, params)
                for query, point, retrieve in zip(queries, query_points, retrieved)
            ])
        return retrieved

    async def _selection_pools(
        self,
        query_point: Point,
        result: LocalSearchResult,
        params: LocalParams,
    ) -> Dict[str, List[Any]]:
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
            "entities": result.entities[:params.select_pool_entities],
            "relations": relations[:params.select_pool_relations],
            "chunks": chunks[:params.select_pool_chunks],
        }

    async def _select_cosine(
        self,
        query: str,
        query_point: Point,
        retrieve: LocalSearchRetrieve,
        params: LocalParams,
    ) -> None:
        """
        Keep the most query-similar candidates of each section, without a decision model.

        Uses the same pools as :meth:`_select_pointwise`, so the two selectors
        differ only in how candidates are chosen from them.

        :param query: Input query string (unused; kept for a uniform signature).
        :param query_point: Query vector built by the retriever.
        :param retrieve: Assembled local context of the query.
        :param params: Selection parameters.
        """
        result = retrieve.result
        pools = await self._selection_pools(query_point, result, params)
        result.entities = pools["entities"][:params.select_top_entities]
        result.relations = pools["relations"][:params.select_top_relations]
        result.chunks = pools["chunks"][:params.select_top_chunks]
        retrieve.metrics["selection"] = {
            name: {"pool": len(pools[name]), "kept": len(kept), "n_calls": 0}
            for name, kept in (
                ("entities", result.entities),
                ("relations", result.relations),
                ("chunks", result.chunks),
            )
        }

    async def _select_pointwise(
        self,
        query: str,
        query_point: Point,
        retrieve: LocalSearchRetrieve,
        params: LocalParams,
    ) -> None:
        """
        Keep only the entities, relations and chunks the decision model judges useful.

        Relations and chunks are first ordered by cosine similarity to the query
        so that the judged pools hold the most similar candidates. The selection
        replaces the sections of ``retrieve.result`` in place and is reported in
        ``retrieve.metrics["selection"]``.

        :param query: Input query string.
        :param query_point: Query vector built by the retriever.
        :param retrieve: Assembled local context of the query.
        :param params: Selection parameters.
        """
        assert self.decision_model is not None
        result = retrieve.result
        pools = await self._selection_pools(query_point, result, params)
        entity_out, relation_out, chunk_out = await asyncio.gather(
            self.decision_model.a_score_pointwise(
                query, pools["entities"], _format_entity, threshold=params.select_threshold,
            ),
            self.decision_model.a_score_pointwise(
                query, pools["relations"], _format_relation, threshold=params.select_threshold,
            ),
            self.decision_model.a_score_pointwise(
                query, pools["chunks"], _format_chunk, threshold=params.select_threshold,
            ),
        )
        result.entities = _keep_with_floor(entity_out, pools["entities"], params.select_floor)
        result.relations = _keep_with_floor(relation_out, pools["relations"], params.select_floor)
        result.chunks = _keep_with_floor(chunk_out, pools["chunks"], params.select_floor)
        retrieve.metrics["selection"] = {
            name: {
                "pool": len(pools[name]),
                "kept": len(kept),
                "passed": len(output.selected),
                "n_calls": output.n_calls,
                "degraded": output.degraded_count,
                "probabilities": [round(c.probability, 4) for c in output.scored],
            }
            for name, output, kept in (
                ("entities", entity_out, result.entities),
                ("relations", relation_out, result.relations),
                ("chunks", chunk_out, result.chunks),
            )
        }

    async def _rank_entities(
        self,
        query: str,
        entities: List[Entity],
        rerank_top_k: int | None = None,
    ) -> List[Entity]:
        """
        Select the entities most relevant to the query from those retrieved.

        The chosen entities become the anchors around which the rest of the local
        context is gathered.

        :param query: Input query string.
        :param entities: Candidate entities retrieved for the query.
        :param rerank_top_k: Keep at most this many; ``None`` keeps all.
        :return: The relevant entities, most relevant first.
        """
        entities = await _rerank_items(
            query,
            entities,
            lambda entity: f"{entity.entity_name}\n{entity.entity_type}\n{entity.description}",
            self.reranker,
        )
        if self.reranker is not None and rerank_top_k is not None:
            entities = entities[:rerank_top_k]
        return entities

    async def _assemble(
        self,
        query: str,
        entities: List[Entity],
        entity_hits: List[EmbeddingHit],
        ranked_entities: List[Entity],
        edges_by_entity: Dict[str, List[Relation]],
        neighbor_chunks: Dict[str, List[str]],
    ) -> LocalSearchRetrieve:
        """
        Gather the local context connected to a query's anchor entities.

        Collects the relations those entities participate in, the text chunks they
        were extracted from, the summaries of the communities they belong to, and
        the documents they originate from — each ordered by relevance to the query —
        and reports the entities' relevance scores.

        :param query: Input query string.
        :param entities: Retrieved entities carrying the reported relevance scores.
        :param entity_hits: Relevance scores aligned with ``entities``.
        :param ranked_entities: The anchor entities selected for this query.
        :param edges_by_entity: Incident edges available for these entities.
        :param neighbor_chunks: Neighbor source chunks available for these entities.
        :return: A :class:`LocalSearchRetrieve` with the query's local context and metrics.
        """
        entity_scores_by_id = {
            entity.id: hit.distance
            for entity, hit in zip(entities, entity_hits)
            if entity and entity.id
        }
        entities = ranked_entities

        # Derive related items from the reduced entity set (document ids are a
        # set union, independent of entity order).
        relations, relevant_chunks, summaries, documents_id = await asyncio.gather(
            _find_most_related_edges_from_entities(entities, edges_by_entity),
            _find_most_related_text_unit_from_entities(
                entities, self.knowledge_graph, edges_by_entity, neighbor_chunks
            ),
            _find_most_related_community_from_entities(entities, self.knowledge_graph),
            _find_documents_id(entities),
        )
        relations = [relation for relation in relations if relation is not None]
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]
        summaries = [summary for summary in summaries if summary is not None]

        # 4. Rerank the derived items.
        relations, summaries, relevant_chunks = await asyncio.gather(
            _rerank_items(
                query,
                relations,
                lambda relation: (
                    f"{relation.subject_name}\n{relation.relation_type}\n"
                    f"{relation.object_name}\n{relation.description}"
                ),
                self.reranker,
            ),
            _rerank_items(
                query,
                summaries,
                lambda community_summary: community_summary.summary,
                self.reranker,
            ),
            _rerank_items(
                query,
                relevant_chunks,
                lambda chunk: chunk.content,
                self.reranker,
            ),
        )

        return LocalSearchRetrieve(
            query=query,
            result=LocalSearchResult(
                entities=entities,
                relations=relations,
                summaries=summaries,
                chunks=relevant_chunks,
                documents_id=documents_id,
            ),
            metrics={
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.entity_name,
                        "rank": idx,
                        "relevance_score": entity_scores_by_id.get(entity.id),
                    }
                    for idx, entity in enumerate(entities)
                ],
            },
        )

    def _render_answer_messages(
        self,
        query: str,
        context: LocalSearchRetrieve,
        use_summary: bool,
        use_chunks: bool,
    ) -> ChatMessages:
        """
        Build the final-answer conversation from a retrieved local context.

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

        truncated_context: str = self.truncation(str(context))
        instruction: RAGUInstruction = self.get_prompt("local_search")

        rendered_conversations: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        return rendered_conversations[0]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: LocalParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Answer each query from its local knowledge-graph context.

        For every query, retrieves the entities most relevant to it together with
        their surrounding relations, chunks and community summaries, and produces a
        natural-language answer grounded in that context.

        :param queries: User queries in natural language.
        :param params: Query parameters. When ``None``, defaults to :class:`LocalParams`.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        """
        if not queries:
            return []

        params = params or LocalParams()
        contexts = await self.batch_search(queries, params)

        conversations = [
            self._render_answer_messages(query, context, params.use_summary, params.use_chunks).to_openai()
            for query, context in zip(queries, contexts)
        ]

        instruction: RAGUInstruction = self.get_prompt("local_search")
        answers = await self.llm.batch_chat_completion(
            conversations,
            output_schema=instruction.pydantic_model,
            desc="LocalSearch batch query",
        )

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
        params: LocalParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute local graph RAG and stream the final plain-text answer.

        :param query: User query in natural language.
        :param params: Query parameters. When ``None``, defaults to
            :class:`LocalParams`.
        :returns: Async iterator of text deltas with the retrieval context.
        """
        params = params or LocalParams()
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


async def _order_by_query_similarity(
    query_point: Point,
    items: List[Any],
    vector_db: BaseVectorStorage,
) -> List[Any]:
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
    scored: List[tuple[float, int]] = []
    missing: List[int] = []
    for i, point in enumerate(points):
        if point is None or point.dense_embedding is None:
            missing.append(i)
            continue
        vector = np.asarray(point.dense_embedding, dtype=np.float32)
        scored.append((float(vector @ query / (np.linalg.norm(vector) or 1.0)), i))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [items[i] for _, i in scored] + [items[i] for i in missing]


def _keep_with_floor(output: DecisionOutput[Any], pool: List[Any], floor: int) -> List[Any]:
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
