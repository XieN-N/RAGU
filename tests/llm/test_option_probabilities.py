"""Integration tests: CachedAsyncOpenAI.option_probabilities against OpenAIMockServer."""
import math
from collections.abc import Generator
from typing import Any

import pytest

from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.testing.openai_mock_server import OpenAIMockServer

_MSG: list[Any] = [{'role': 'user', 'content': 'Is it relevant? Answer Y or N:'}]


@pytest.fixture
def server() -> Generator[OpenAIMockServer, None, None]:
    srv = OpenAIMockServer()
    srv.start()
    yield srv
    srv.stop()


async def test_masses_sum_whitespace_variants(server: OpenAIMockServer) -> None:
    # "Y" and " Y" are different tokens but the same option
    server.set_top_logprobs([('Y', -0.4), (' Y', -2.5), ('N', -2.0), ('A', -3.0)])
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    masses = await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['Y', 'N'],
    )
    assert masses['Y'] == pytest.approx(math.exp(-0.4) + math.exp(-2.5))
    assert masses['N'] == pytest.approx(math.exp(-2.0))


async def test_missing_option_gets_zero(server: OpenAIMockServer) -> None:
    # top_logprobs=1 leaves only the emitted token
    server.set_top_logprobs([('Y', -0.1), ('N', -2.3)])
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    masses = await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['Y', 'N'], top_logprobs=1,
    )
    assert masses['Y'] == pytest.approx(math.exp(-0.1))
    assert masses['N'] == 0.0


async def test_request_is_single_greedy_token(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(
        base_url=server.base_url, api_key='mock', max_completion_tokens=512,
    )
    await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['Y', 'N'],
        top_logprobs=5, logit_bias={56: 100, 45: 100},
        extra_body={'allowed_token_ids': [56, 45]},
    )
    body = server.last_chat_body
    assert body is not None
    # the instance-level max_completion_tokens must not leak into decisions
    assert body['max_completion_tokens'] == 1
    assert body['temperature'] == 0.0
    assert body['logprobs'] is True
    assert body['top_logprobs'] == 5
    assert body['logit_bias'] == {'56': 100, '45': 100}
    assert body['allowed_token_ids'] == [56, 45]


async def test_unsupported_kwargs_raise(server: OpenAIMockServer) -> None:
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')
    with pytest.raises(TypeError, match='Unsupported option_probabilities arguments'):
        await client.option_probabilities(
            model_name='mock', conversation=_MSG, options=['Y'], temperature=0.7,
        )


async def test_results_are_cached(server: OpenAIMockServer) -> None:
    server.set_top_logprobs([('Y', -0.1), ('N', -2.3)])
    client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock', cache={})
    first = await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['Y', 'N'],
    )
    # a changed server answer must not be seen: the second call is a cache hit
    server.set_top_logprobs([('N', -0.1), ('Y', -2.3)])
    second = await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['Y', 'N'],
    )
    assert second == first
    # a different option set is a different cache key
    other = await client.option_probabilities(
        model_name='mock', conversation=_MSG, options=['N', 'Y'],
    )
    assert other['N'] == pytest.approx(math.exp(-0.1))


async def test_llm_openai_forwards(server: OpenAIMockServer) -> None:
    server.set_top_logprobs([('A', -0.2), ('B', -1.9)])
    llm = LLMOpenAI(
        CachedAsyncOpenAI(base_url=server.base_url, api_key='mock'),
        'mock-model',
        temperature=0.9,  # generation default; must not reach the decision call
    )
    masses = await llm.option_probabilities(_MSG, ['A', 'B'])
    assert masses['A'] == pytest.approx(math.exp(-0.2))
    assert server.last_chat_body is not None
    assert server.last_chat_body['model'] == 'mock-model'
    assert server.last_chat_body['temperature'] == 0.0
