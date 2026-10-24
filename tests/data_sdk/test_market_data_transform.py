"""Unit tests for Yahoo Finance market-data ingestion helpers."""

import unittest
from datetime import UTC, date, datetime
from unittest.mock import patch

import ml_platform.data_sdk.dataset_transforms as dataset_transforms
from ml_platform.data_sdk.exceptions import YahooFinancePayloadError


class YahooChartRowsTest(unittest.TestCase):
    """Validate Yahoo response parsing before Spark is involved."""

    def _payload(self) -> dict[str, object]:
        """Return a minimal valid Yahoo chart response with one incomplete bar."""
        return {
            "chart": {
                "result": [
                    {
                        "timestamp": [1767364200, 1767367800],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0, None],
                                    "high": [11.0, 12.0],
                                    "low": [9.0, 10.0],
                                    "close": [10.5, 11.5],
                                    "volume": [100, 200],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

    def test_rows_convert_complete_bars_to_utc_tuples(self) -> None:
        """Incomplete upstream bars are excluded from typed output rows."""
        rows = dataset_transforms._market_data._yahoo_chart_rows(
            self._payload(), "AAA"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].timestamp, datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
        )
        self.assertEqual(rows[0].symbol, "AAA")
        self.assertEqual(rows[0].open_price, 10.0)
        self.assertEqual(rows[0].high_price, 11.0)
        self.assertEqual(rows[0].low_price, 9.0)
        self.assertEqual(rows[0].close_price, 10.5)
        self.assertEqual(rows[0].volume, 100)

    def test_malformed_payloads_raise_domain_errors(self) -> None:
        """Missing chart data, provider errors, and inconsistent arrays are rejected."""
        cases = [
            {},
            {"chart": {"error": "unknown symbol"}},
            {"chart": {"result": [{}]}},
            {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1],
                            "indicators": {"quote": [{"open": []}]},
                        }
                    ]
                }
            },
        ]
        for payload in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaises(YahooFinancePayloadError),
            ):
                dataset_transforms._market_data._yahoo_chart_rows(
                    payload, "AAA"
                )

    def test_no_provider_results_returns_no_rows(self) -> None:
        """An empty Yahoo result is distinct from a malformed response."""
        self.assertEqual(
            dataset_transforms._market_data._yahoo_chart_rows(
                {"chart": {"result": []}}, "AAA"
            ),
            [],
        )


class MarketDataRangeTest(unittest.TestCase):
    """Verify date-range compaction used to limit provider calls."""

    def test_contiguous_ranges_split_gaps_and_use_exclusive_end_dates(
        self,
    ) -> None:
        """Adjacent dates share one half-open Yahoo request range."""
        self.assertEqual(
            dataset_transforms._market_data._contiguous_date_ranges(
                [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4)]
            ),
            [
                (date(2026, 1, 1), date(2026, 1, 3)),
                (date(2026, 1, 4), date(2026, 1, 5)),
            ],
        )
        self.assertEqual(
            dataset_transforms._market_data._contiguous_date_ranges([]), []
        )


class YahooHttpClientTest(unittest.TestCase):
    """Verify the transform's provider client preserves its request contract."""

    @patch("ml_platform.data_sdk.dataset_transforms._market_data.httpx2.get")
    def test_request_chart_uses_timeout_and_user_agent(
        self, request: object
    ) -> None:
        """Yahoo requests identify the client and fail on non-success responses."""
        response = request.return_value
        response.json.return_value = {"chart": {"result": []}}

        payload = dataset_transforms._market_data._request_yahoo_chart(
            "AAPL", date(2026, 1, 2), date(2026, 1, 3)
        )

        self.assertEqual(payload, {"chart": {"result": []}})
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(request.call_args.kwargs["timeout"], 30)
        self.assertEqual(
            request.call_args.kwargs["headers"],
            {"User-Agent": "ml-platform/0.1"},
        )
