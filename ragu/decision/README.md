# Module: ragu.decision

## Role in RAGU Pipeline

`ragu.decision` provides **single-token decision models**: an LLM answers a
closed question with exactly one output token, and the answer distribution is
read from that token's `top_logprobs` instead of parsing generated text.
Search engines and builder modules use them to select, rank and route cheaply:

```text
query + candidates -> DecisionModel -> LLM.option_probabilities -> CachedAsyncOpenAI -> API
```

One request costs a prompt prefill plus one decoded token. The answer can never
fall outside the option set, and the returned probabilities can be thresholded.

## Key Components

### DecisionModel

`DecisionModel(llm, ...)` works on top of any `LLM` that implements
`option_probabilities` (`LLMOpenAI` against OpenAI or vLLM). It can share the
answer-generation `LLM` or use a separate one, e.g. a local vLLM model for
decisions while answers come from an API model.

| Method | Purpose | Requests |
|--------|---------|----------|
| `a_decide(question, options, context="")` | Pick one of 2–20 options (routing, stop/continue) | 1 |
| `a_score_pointwise(query, candidates, format_fn=str, state_context="", threshold=0.5, top_k=None)` | Independent `Y`/`N` relevance per candidate; candidates do not compete for probability mass | one per candidate, concurrent |
| `a_score_listwise(query, candidates, ..., top_k=3, min_prob=0.05, max_none_prob=0.5)` | Up to 19 lettered candidates plus a "none" option `Z`; the first-token distribution is read as a ranking | 1 |

Pointwise scoring is the default choice for selection: the candidate text comes
last in the prompt, so on vLLM with `--enable-prefix-caching` the shared
query/state prefix is computed once. Listwise scoring is cheaper on APIs
without prefix caching but candidates share one unit of probability and are
subject to position bias.

Constructor parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm` | Backend `LLM` (required) | — |
| `constraint` | `"none"` reads `top_logprobs` as is; `"logit_bias"` boosts label tokens (OpenAI); `"allowed_token_ids"` restricts decoding to label tokens via `extra_body` (vLLM) | `"none"` |
| `tokenizer_backend` | `"tiktoken"` or `"local"` (HuggingFace); used only when `constraint != "none"` | `"tiktoken"` |
| `tokenizer_model` | Model or encoding name, e.g. `gpt-4o-mini`, `o200k_base`, `bond005/meno-lite-0.1` | `"gpt-4o-mini"` |
| `top_logprobs` | First-token alternatives to request (OpenAI allows at most 20) | `20` |
| `bias` | Logit bias for label tokens when `constraint="logit_bias"` | `100` |

Token ids are collected for every label both bare and with a leading space
(`"Y"` and `" Y"` are different BPE tokens); a label that is not a single token
raises `ValueError`.

### Results

- `SingleDecision`: `choice` (option text), `probability`, `distribution`
  (sums to 1), `entropy` (nats), `degraded`.
- `DecisionOutput[T]`: `selected` (best first, cut to `top_k`), `scored`
  (`ScoredCandidate[T]` sorted by probability), `n_calls`,
  `none_probability` (listwise), `degraded_count`.

A result is *degraded* when none of the label tokens appeared in
`top_logprobs`; the distribution is then uniform and a warning is logged.

## Usage

```python
from ragu.decision import DecisionModel
from ragu.models import CachedAsyncOpenAI, LLMOpenAI

client = CachedAsyncOpenAI(base_url="https://api.openai.com/v1", api_key="sk-...")
decision_model = DecisionModel(LLMOpenAI(client, "gpt-4o-mini"))

output = await decision_model.a_score_pointwise(
    query="Where did the father of the creator of C work?",
    candidates=relations,
    format_fn=lambda r: f"{r.subject_name} --[{r.relation_type}]--> {r.object_name} | {r.description}",
    threshold=0.5,
)
useful_relations = output.selected

route = await decision_model.a_decide(
    "Which kind of question is this?",
    ["single fact", "multi-hop chain", "summary of a topic", "broad synthesis"],
    context="Where did the father of the creator of C work?",
)
```

## Notes

- Probabilities are model-dependent and not calibrated by construction; tune
  thresholds per model on held-out data.
- Requests go through `CachedAsyncOpenAI.option_probabilities`, so they share
  the client's cache, retries and rate limits. They always use
  `max_completion_tokens=1` and `temperature=0`, regardless of the client's
  `max_completion_tokens` or the `LLMOpenAI` constructor kwargs.
- Reasoning models that spend the first token on a reasoning channel are not
  supported.
- Prompts: `decision_choice`, `decision_pointwise`, `decision_listwise`
  (override with `update_prompt`, see `ragu/common/prompts/README.md`).
