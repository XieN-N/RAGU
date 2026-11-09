# Module: ragu.search_engine

## Role in RAGU Pipeline

`ragu.search_engine` is the query layer. It consumes an already built `KnowledgeGraph`, retrieves context with one or more strategies, and optionally asks an LLM to synthesize an answer.

Pipeline position:

```text
KnowledgeGraph + query -> retrieval context -> LLM answer
```

## Overview

The module separates retrieval strategy from indexing. All engines share `BaseEngine`, the async `search()` for retrieval-only calls, the async `query()` for retrieval plus complete answer generation, and `stream_query()` for retrieval plus streamed text generation where supported. For many queries at once, every engine also exposes async `batch_search()` / `batch_query()`, which run retrieval concurrently and — for the leaf engines (Local, Decision, Naive, Global) — route answer generation through a single `llm.batch_chat_completion` call. `search()`, `query()`, `batch_search()`, and `batch_query()` are coroutines (await them); `stream_query()` returns an async iterator. The `a_search` / `a_query` names are kept as backward-compatible async aliases that delegate to `search` / `query`.

## Key Components

### BaseEngine

Abstract base for query engines.

- Purpose: stores LLM and context truncation settings.
- Important methods: `search`, `query`, `stream_query`.
- Batch methods (all async): `batch_search`, `batch_query`.
  They take a `list[str]` of queries and return a result list **aligned by position** with the
  input. They are fail-fast: the first failing query aborts the whole batch and raises.
- Backward-compatible async aliases: `a_search`, `a_query` delegate to `search` / `query`.


### SearchEngineRetrieve

Base dataclass for retrieval-only results.

- Purpose: carries `query`, engine-specific `result`, and `metrics`.
- Important method: `to_text()`.

```python
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(
    query="What is RAGU?",
    result=NaiveSearchResult(),
    metrics={"chunks": []},
)

print(retrieval.to_text())
```

### SearchEngineResponse

Generated answer container.

- Purpose: carries `query`, `response`, `retrieval`, and optional `payload`.

```python
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.naive_search import NaiveSearchResult, NaiveSearchRetrieve

retrieval = NaiveSearchRetrieve(query="What is RAGU?", result=NaiveSearchResult())
response = SearchEngineResponse(
    query="What is RAGU?",
    response="RAGU is a GraphRAG engine.",
    retrieval=retrieval,
)

print(str(response))
```

### SearchEngineStreamEvent

Incremental generated-answer event returned by `stream_query()`.

- Purpose: carries `query`, the already retrieved `retrieval` context, generated
  text `delta`, and optional `payload`.
- Streaming is text-only; use `query()` for structured Pydantic responses.

```python
from ragu import SearchEngineStreamEvent

async for event in engine.stream_query("What is RAGU?"):
    assert isinstance(event, SearchEngineStreamEvent)
    print(event.delta, end="")
```

### NaiveSearchEngine

Chunk-vector RAG.

- Purpose: retrieve chunk vectors directly, optionally rerank, then answer.
- Best for: document QA when graph extraction is not needed.
- Uses: `GraphRetriever.query_chunks`.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

### LocalSearchEngine

Graph-neighborhood RAG.

- Purpose: retrieve relevant entities, then collect related relations, chunks, and community summaries.
- Best for: entity-centric questions and local factual neighborhoods.
- Uses: entity vector DB, graph edges, chunk KV, community summary KV.

```python
from ragu import KnowledgeGraph, LocalSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

print(engine.language)
```

### DecisionSearchEngine

Graph-neighborhood RAG in which a decision model chooses the context.

- Purpose: start from the neighbourhood `LocalSearchEngine` would retrieve, keep what a
  single-token decision model (see [`ragu/decision/README.md`](../decision/README.md))
  accepts, and optionally walk further hops from the relations it kept.
- Best for: multi-hop questions, and whenever a shorter context matters more than
  one extra request per candidate.
- Requires: `decision_model`, which has no default; without one use `LocalSearchEngine`.
- Works under `QueryPlanEngine` and `MixSearchEngine` like any other engine. The answer
  prompt is `local_search`.

```python
from ragu import DecisionModel, DecisionSearchEngine, KnowledgeGraph
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.search_engine.decision_search import DecisionParams

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = DecisionSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder,
                              decision_model=DecisionModel(llm))
params = DecisionParams(select_hops=2, select_pool_chunks=30)

print(params.policy)
```

### GlobalSearchEngine

Community-summary RAG.

- Purpose: evaluate all community summaries for query relevance and synthesize a global answer.
- Best for: broad questions that require corpus-level themes.
- Requires: community summaries built by `KnowledgeGraph`.

```python
from ragu import GlobalSearchEngine, KnowledgeGraph
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(llm=llm, embedder=embedder)
engine = GlobalSearchEngine(llm=llm, knowledge_graph=graph)

print(engine.language)
```

### MixSearchEngine

Ensemble engine.

- Purpose: run child engines and synthesize combined context or combined answers.
- Important parameter: `ensemble_responses`.

```python
from ragu import BuilderArguments, KnowledgeGraph, MixSearchEngine, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
naive = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
mix = MixSearchEngine(llm=llm, engines=[naive])

print(len(mix.engines))
```

### QueryPlanEngine

Wraps any search engine and decomposes a complex query into dependent subqueries: each subquery is executed via the wrapped engine, intermediate answers are fed into subsequent subqueries, and the partial answers are synthesized into a final response. Useful for multi-hop questions that no single retrieval strategy can answer directly.

`stream_query()` is supported when the wrapped engine supports it. Dependency
subqueries are executed normally so their complete answers can be used for
rewriting; only the final sink subquery answer is streamed.

```python
from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine, QueryPlanEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="dummy-api-token")
llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
embedder = EmbedderOpenAI(client=client, model_name="text-embedding-3-small", dim=1536)
graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
)
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
planner = QueryPlanEngine(engine)

print(planner.get_prompt("query_decomposition").description)
```

## Data Flow

Input: query string, `KnowledgeGraph`, dense embedder, optional sparse embedder and reranker.

Output:

- `SearchEngineRetrieve` from `search`
- `SearchEngineResponse` from `query`
- `SearchEngineStreamEvent` deltas from `stream_query`

Used by: applications that need GraphRAG answers, retrieval diagnostics, or mixed retrieval strategies.

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu import BuilderArguments, KnowledgeGraph, NaiveSearchEngine
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(["Python is a programming language."])

    engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    retrieval = await engine.search("What is Python?")
    print(retrieval.result.chunks[0].content)


asyncio.run(main())
```

### Example 2 - Pipeline usage

```python
import asyncio

from ragu import BuilderArguments, GlobalSearchEngine, KnowledgeGraph, LocalSearchEngine, MixSearchEngine
from ragu.graph.types import CommunitySummary, Entity, Relation
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


async def main():
    client = CachedAsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key="dummy-api-token",
    )
    llm = LLMOpenAI(client=client, model_name="gpt-4o-mini")
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=1536,
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        builder_settings=BuilderArguments(make_community_summary=False),
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        subject_id=guido.id,
        object_id=python.id,
        subject_name=guido.entity_name,
        object_name=python.entity_name,
        relation_type="CREATED",
        description="Guido van Rossum created Python.",
        source_chunk_id=["chunk-1"],
    )
    await graph.upsert_entities([python, guido])
    await graph.upsert_relations([relation])
    await graph.upsert_summaries([CommunitySummary(id="com-1", summary="Python has a creator.")])

    local = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    global_ = GlobalSearchEngine(llm=llm, knowledge_graph=graph)
    mix = MixSearchEngine(llm=llm, engines=[local, global_])

    response = await mix.query("Summarize the main people and organizations.")
    print(response.response)

    async for event in mix.stream_query("Summarize the main people and organizations."):
        print(event.delta, end="")


asyncio.run(main())
```

### Example 3 - Override a search engine instruction

Every search engine inherits `RaguGenerativeModule`, so `get_prompt` /
`update_prompt` work out of the box:

```python
engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
engine.get_prompt("naive_search")            # inspect current instruction
engine.update_prompt("naive_search", my_instruction)  # override for this instance only
```

For a full custom-message example see
[`ragu/common/prompts/README.md`](../common/prompts/README.md). Do **not** edit
`DEFAULT_PROMPT_TEMPLATES` directly; `update_prompt` scopes the change to a
single engine instance.

## Integration Points

- LLMs: every `query()` delegates to `batch_query()`; leaf `batch_query()` implementations render one prompt per query and issue a single `llm.batch_chat_completion`. `stream_query()` renders one final-answer prompt and calls `llm.stream_chat_completion`.
- Embedders: local and naive search encode query text through `GraphRetriever`.
- Sparse embedders: local and naive search can issue hybrid dense+sparse vector queries when storage supports sparse vectors.
- Qdrant: hybrid search is handled by vector storage, including Qdrant prefetch/fusion in `QdrantVectorDBStorage`.
- Other modules: search engines read graph artifacts through `KnowledgeGraph.index`.

## Configuration

Shared engine parameters:

- `language`: prompt language, defaulting to `Settings.language`.

Engine token limits are configured centrally via `Settings.llm_context_token_limit`,
`Settings.tokenizer_llm_backend`, and `Settings.tokenizer_llm_name`.

For per-instance control (e.g. several engines with different LLMs / context
windows in the same process), every engine also accepts these constructor
parameters, each defaulting to `None`:

- `max_context_length` (`int | None`) — falls back to `Settings.llm_context_token_limit`.
- `tokenizer_backend` (`Literal["tiktoken", "local"] | None`) — falls back to `Settings.tokenizer_llm_backend`.
- `tokenizer_model` (`str | None`) — falls back to `Settings.tokenizer_llm_name`.

```python
engine = LocalSearchEngine(
    llm=llm, knowledge_graph=kg, embedder=emb,
    max_context_length=16_000,
    tokenizer_model="gpt-4o",
)
```

> Note: an override on `MixSearchEngine` affects **only** its own final-context
> truncation and is **not** propagated to the child engines — each child keeps
> the tokenizer configuration it was constructed with.

Retrieval parameters:

- `top_k`: initial result count for local and naive search.
- `rerank_top_k`: final chunk count for `NaiveSearchEngine` when a reranker is configured.
- `use_summary` and `use_chunks`: toggles for `LocalSearchEngine.query` and `DecisionSearchEngine.query`.
- `allow_partial_failures`: controls `MixSearchEngine` child-engine failures.

Stage timings written by `LocalSearchEngine`: `retrieve.metrics["timing"]` holds
seconds per stage: `retrieve` (entity search and context assembly) and `generate`
(answer LLM call, set by `batch_query`). Stages are measured per batch.

Parameters of `DecisionSearchEngine` (`DecisionParams`) repeat `top_k`, `rerank_top_k`,
`use_summary` and `use_chunks`, and add the walk described below.

- Pools: the first `select_pool_entities` entities (retrieval order) and the
  `select_pool_relations` relations and `select_pool_chunks` chunks most similar
  to the query (cosine against their stored vectors). Defaults: 20 / 30 / 10.
- Judgement: the decision model answers a `Y`/`N` relevance question per pooled
  candidate; candidates with `P(Y) >= select_threshold` (default `0.5`) are kept,
  topped up to `select_floor` (default `3`) per section in pool order. One request
  per candidate. With a strict model the floor often decides most of the context:
  on MuSiQue with `gpt-4o-mini` the median question had 1 entity, 2 relations and
  1 chunk passing, so the rest of the floor came from cosine order.
- `select_hops` (default `1`): with `n > 1` the selection walks `n - 1` more hops.
  The unvisited endpoints of relations that passed on the previous hop form the
  frontier; their entities, relations and chunks are pooled by cosine similarity to
  the query plus the facts kept so far, judged with those facts as `state_context`,
  and the passing ones are added to the context. A hop with no passing relation ends
  the walk. Only relations that passed the threshold extend the frontier, not those
  added by the floor, so when nothing passes on the seed hop there is no further
  hop. Seed-neighbourhood items left outside the pools count as seen and are not
  reconsidered on later hops.
- `select_state_max_chars` (default `2000`): maximum length of the kept-facts text
  used as `state_context` and in the hop query vector.
- `select_keep_seed_context` (default `False`): keep the full seed context instead
  of replacing it with the selection. The decision model then only chooses the
  direction of further hops (`select_hops > 1`) and the facts they append after the
  seed context.

Strategy options. A value outside the list raises `ValueError`. Options marked
`best_first` need `policy="best_first"`.

| Option | Values | Meaning |
|---|---|---|
| `seeds` | `entities`, `entities+facts`, `entities+facts+names` | where the walk starts: entities similar to the query, plus the most similar relations (`seed_facts`), plus entities named in the query |
| `policy` | `threshold`, `one_round`, `best_first` | keep what passes `select_threshold` and expand it; judge the two-hop neighbourhood once; expand the most probable relations until a budget runs out |
| `state` (`best_first`) | `facts`, `chain` | what the model is shown as gathered so far: the kept facts or the relations leading to the candidate |
| `next_query` (`best_first`) | `blob`, `subquestion` | what further candidates are ranked against: the query plus the state, or a question the LLM writes for the next step, also searched over the whole index |
| `stop` (`best_first`) | `hops`, `budget`, `sufficient` | when the walk ends |
| `context_order` | `sections`, `ranked`, `chunks_first`, `flat` | how the kept items are arranged |
| `tie_break` (`best_first`) | `none`, `cosine` | order of items with nearly equal `P(Y)`; with `cosine`, by `P(Y) + tie_weight * cosine` |

`best_first` budgets: `max_decisions` (default 150), `max_context_tokens` (default 3000)
and `min_probability` (default `select_threshold`). `select_beam` relations are expanded
per round, chains are at most `select_max_depth` long, and every expanded relation adds
at most `branch_pool_relations` relations and `branch_pool_chunks` chunks. The walk does
not depend on how long storage calls take: expanded branches never judge the same
candidate twice, and the earlier relation in the queue gets it. `log_decisions=True`
puts every judged item into `metrics["decisions"]`.

Configuration that gave the largest gain on MuSiQue:

```python
DecisionParams(
    policy="best_first", stop="budget", next_query="subquestion", select_max_depth=3,
    min_probability=0, max_context_tokens=3000, context_order="flat", select_pool_chunks=30,
)
```

The engine does not change the answer prompt. A prompt that forbids abstaining or asks
for a short reasoning is set with `update_prompt("local_search", ...)`.

Diagnostics written by `DecisionSearchEngine`:

- `retrieve.metrics["entities"]`: the retrieved entities before selection;
  `retrieve.result` holds what goes into the answer prompt.
- `retrieve.metrics["selection"]`: per-section pool, kept and passed counts, requests,
  degraded answers and probabilities of the seed hop.
- `retrieve.metrics["selection_hops"]`: per-hop statistics when `select_hops > 1`.
- `retrieve.metrics["decision_usage"]`: `calls` and `prompt_chars` of all
  decision-model requests for the query.
- `retrieve.metrics["timing"]`: seconds per stage: `retrieve`, `select` (decision
  model, including further hops), `hops` (further hops only) and `generate` (answer
  LLM call, set by `batch_query`). Stages are measured per batch.

## Dependencies

Internal:

- `ragu.graph.KnowledgeGraph`
- `ragu.graph.GraphRetriever`
- `ragu.models`
- `ragu.common.prompts`
- `ragu.decision`
- `ragu.utils.token_truncation`

External:

- `jinja2`
- `pydantic`
- `typing_extensions`

## Notes / Pitfalls

- `GlobalSearchEngine` needs community summaries; build with `make_community_summary=True` or call `reindex_community()`.
- `LocalSearchEngine` and `DecisionSearchEngine` start from entity vector search, so they need entity vectors, not just chunk vectors.
- `NaiveSearchEngine` works with `build_only_vector_context=True`.
- `MixSearchEngine` requires at least one child engine.
- `search()` is useful for debugging retrieval before involving the LLM.
