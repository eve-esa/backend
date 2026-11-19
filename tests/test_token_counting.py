"""Tokenizer-backed message counts, the per-message cache, and a missing file."""

from langchain_core.messages import HumanMessage

from src.core import token_counting

import pytest

pytestmark = pytest.mark.no_db


def _tokenizer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "aa": 1, "bb": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return tokenizer, path


def test_the_counter_matches_the_tokenizer_and_caches(tmp_path, monkeypatch):
    tokenizer, path = _tokenizer(tmp_path)
    monkeypatch.setenv("MODEL_TOKENIZERS", '{"main": "%s"}' % path)
    token_counting.clear_tokenizer_cache_for_tests()
    counter = token_counting.counter_for("main")
    text = "aa bb"
    expected = len(tokenizer.encode(text, add_special_tokens=False).ids) + 3
    assert counter([HumanMessage(text)]) == expected

    def _fail(*_args, **_kwargs):
        raise AssertionError("the second call must use the cache")

    counter._tokenizer.encode_batch = _fail
    assert counter([HumanMessage(text)]) == expected
    token_counting.clear_tokenizer_cache_for_tests()


def test_a_missing_tokenizer_file_returns_none(monkeypatch):
    monkeypatch.setenv("MODEL_TOKENIZERS", '{"main": "/no/such/tokenizer.json"}')
    token_counting.clear_tokenizer_cache_for_tests()
    assert token_counting.counter_for("main") is None
    assert token_counting.counter_for("fallback") is None
    token_counting.clear_tokenizer_cache_for_tests()


def test_an_unknown_model_keeps_the_estimate(monkeypatch):
    monkeypatch.delenv("MODEL_TOKENIZERS", raising=False)
    token_counting.clear_tokenizer_cache_for_tests()
    assert token_counting.counter_for("fallback") is None
    token_counting.clear_tokenizer_cache_for_tests()
