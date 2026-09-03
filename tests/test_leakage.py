"""Leakage guard tests."""

from __future__ import annotations

import pandas as pd
import pytest

from vulnprio import leakage
from vulnprio.data import schema


class TestScrubbing:
    def test_removes_leaky_phrases(self):
        text = "A flaw that has been exploited in the wild by ransomware operators."
        scrubbed = leakage.scrub_text(text)
        assert "exploited in the wild" not in scrubbed.lower()
        assert "ransomware" not in scrubbed.lower()

    def test_is_case_insensitive(self):
        assert "ACTIVELY EXPLOITED" not in leakage.scrub_text("An ACTIVELY EXPLOITED bug.")

    def test_preserves_non_leaky_content(self):
        text = "A buffer overflow in the parser allows remote code execution."
        assert leakage.scrub_text(text) == text

    def test_collapses_whitespace_left_behind(self):
        assert "  " not in leakage.scrub_text("A zero-day in the kernel.")

    def test_series_handles_missing_values(self):
        out = leakage.scrub_series(pd.Series(["actively exploited bug", None]))
        assert out.iloc[1] == ""


class TestForbiddenColumns:
    def test_rejects_the_label(self):
        frame = pd.DataFrame({"feature": [1], schema.LABEL: [1]})
        with pytest.raises(leakage.LeakageError, match=schema.LABEL):
            leakage.assert_no_forbidden_columns(frame)

    def test_rejects_the_kev_date(self):
        frame = pd.DataFrame({"feature": [1], schema.LABEL_DATE: [pd.Timestamp("2023-01-01")]})
        with pytest.raises(leakage.LeakageError):
            leakage.assert_no_forbidden_columns(frame)

    def test_accepts_a_clean_matrix(self):
        leakage.assert_no_forbidden_columns(pd.DataFrame({"cvss_base_score": [9.8]}))


class TestTemporalOrder:
    def _frame(self, dates):
        return pd.DataFrame({schema.PUBLISHED: pd.to_datetime(dates)})

    def test_accepts_correct_ordering(self):
        leakage.assert_temporal_order(
            self._frame(["2020-01-01", "2021-01-01"]), self._frame(["2022-01-01"])
        )

    def test_rejects_overlap(self):
        with pytest.raises(leakage.LeakageError, match="Temporal leakage"):
            leakage.assert_temporal_order(
                self._frame(["2020-01-01", "2023-01-01"]), self._frame(["2022-01-01"])
            )

    def test_empty_frames_are_a_no_op(self):
        leakage.assert_temporal_order(self._frame([]), self._frame(["2022-01-01"]))


class TestAudit:
    def test_counts_leaky_descriptions(self):
        frame = pd.DataFrame(
            {
                schema.DESCRIPTION: [
                    "actively exploited in production",
                    "a normal buffer overflow",
                    "used by ransomware groups",
                ]
            }
        )
        audit = leakage.leakage_audit(frame)
        assert audit.get("actively exploited") == 1
        assert audit.get("ransomware") == 1
        assert "a normal buffer overflow" not in audit

    def test_returns_empty_without_a_description_column(self):
        assert leakage.leakage_audit(pd.DataFrame({"x": [1]})) == {}
