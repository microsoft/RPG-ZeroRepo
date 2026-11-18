#!/usr/bin/env python3
"""Tests for model pricing in utils/claude/llm_usage_count.py.

Covers:
  - fallback table: dated ids match, newer model ids do not inherit an older price
  - llm-prices (optional): current model ids resolve to their own prices
  - APICall.cost with the 5-minute / 1-hour cache-write split
"""

import os
import sys

import pytest

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_project_root, "utils", "claude"))

import llm_usage_count
from llm_usage_count import APICall, PRICING, _match_pricing


@pytest.fixture
def table_only(monkeypatch):
    monkeypatch.setattr(llm_usage_count, "llm_prices", None)


class TestFallbackTable:
    def test_dated_ids_match_their_base_entry(self, table_only):
        assert _match_pricing("claude-opus-4-20250514") == PRICING["claude-opus-4"]
        assert _match_pricing("claude-opus-4-5-20251101") == PRICING["claude-opus-4-5"]
        assert _match_pricing("claude-sonnet-4-5-20250929") == PRICING["claude-sonnet-4-5"]

    def test_exact_ids_match(self, table_only):
        assert _match_pricing("claude-opus-4-6") == PRICING["claude-opus-4-6"]
        assert _match_pricing("claude-opus-4.6") == PRICING["claude-opus-4.6"]

    def test_newer_minor_version_does_not_get_older_price(self, table_only):
        # Used to match the "claude-opus-4" prefix and be priced at $15/$75.
        assert _match_pricing("claude-opus-4-7") is None
        assert _match_pricing("claude-opus-4-8") is None

    def test_unknown_model_is_none(self, table_only):
        assert _match_pricing("") is None
        assert _match_pricing("<synthetic>") is None


class TestLlmPrices:
    @pytest.fixture(autouse=True)
    def _require_llm_prices(self):
        pytest.importorskip("llm_prices")

    @pytest.mark.parametrize("model", ["claude-opus-4-7", "claude-opus-4-8", "claude-opus-5"])
    def test_current_opus_ids(self, model):
        assert _match_pricing(model) == (5.0, 6.25, 10.0, 0.5, 25.0)

    def test_haiku_4_5_api_id(self):
        # The API returns claude-haiku-4-5-20251001; the table only had "claude-haiku-4.5".
        assert _match_pricing("claude-haiku-4-5-20251001") == (1.0, 1.25, 2.0, 0.1, 5.0)

    def test_models_missing_from_table_are_priced(self):
        for model in ("claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5-5", "claude-fable-5"):
            assert _match_pricing(model) is not None, model

    def test_bedrock_id(self):
        assert _match_pricing("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == PRICING["claude-sonnet-4-5"]

    def test_existing_entries_unchanged(self):
        for model, key in [
            ("claude-opus-4-6", "claude-opus-4-6"),
            ("claude-opus-4-5-20251101", "claude-opus-4-5"),
            ("claude-opus-4-1-20250805", "claude-opus-4-1"),
            ("claude-sonnet-4-6", "claude-sonnet-4-6"),
            ("claude-sonnet-4-20250514", "claude-sonnet-4"),
        ]:
            assert _match_pricing(model) == pytest.approx(PRICING[key]), model

    def test_call_cost_matches_llm_prices(self):
        import llm_prices

        call = APICall(
            index=1, line_number=1, model="claude-opus-4-7",
            input_tokens=1_000, cache_creation_input_tokens=5_000,
            cache_read_input_tokens=40_000, output_tokens=2_000,
            ephemeral_5m_input_tokens=2_000, ephemeral_1h_input_tokens=3_000,
        )
        expected = llm_prices.lookup_provider("anthropic", "claude-opus-4-7").cost_usage(
            llm_prices.Usage(input=1_000, cache_write_5m=2_000, cache_write_1h=3_000,
                             cache_read=40_000, output=2_000)
        ).total
        assert call.cost() == pytest.approx(expected)
        assert call.cost() == pytest.approx(0.1175)
