"""Tests for the CVSS feature block - the reference implementation.

Also the template: a new block's test file should cover the same four
properties (schema requirements, stable column set across splits, correct
encoding, index preservation).
"""

from __future__ import annotations

import pandas as pd
import pytest

from vulnprio.data import schema
from vulnprio.features.base import available_blocks, get_block
from vulnprio.features.cvss import CVSSBlock


def test_block_is_registered():
    assert "cvss" in available_blocks()
    assert get_block("cvss") is CVSSBlock


def test_produces_one_column_per_cvss_level_plus_score(tiny_cves):
    block = CVSSBlock().fit(tiny_cves)
    out = block.transform(tiny_cves)
    expected = 1 + sum(len(levels) for levels in schema.CVSS_CATEGORICALS.values())
    assert out.shape == (len(tiny_cves), expected)
    assert list(out.columns) == block.feature_names()


def test_column_set_is_identical_across_splits(tiny_cves):
    # Levels come from the CVSS spec, not the observed data, so a split missing
    # a rare level still yields the same matrix shape.
    block = CVSSBlock().fit(tiny_cves.iloc[:6])
    a = block.transform(tiny_cves.iloc[:6])
    b = block.transform(tiny_cves.iloc[6:])
    assert list(a.columns) == list(b.columns)


def test_one_hot_encoding_is_correct(tiny_cves):
    out = CVSSBlock().fit(tiny_cves).transform(tiny_cves)
    assert (out[f"{schema.CVSS_ATTACK_VECTOR}__NETWORK"] == 1).all()
    assert (out[f"{schema.CVSS_ATTACK_VECTOR}__LOCAL"] == 0).all()


def test_index_is_preserved(tiny_cves):
    shuffled = tiny_cves.sample(frac=1, random_state=0)
    out = CVSSBlock().fit(shuffled).transform(shuffled)
    pd.testing.assert_index_equal(out.index, shuffled.index)


def test_score_can_be_excluded(tiny_cves):
    out = CVSSBlock(include_score=False).fit(tiny_cves).transform(tiny_cves)
    assert schema.CVSS_BASE_SCORE not in out.columns


def test_missing_required_column_raises(tiny_cves):
    with pytest.raises(schema.SchemaError):
        CVSSBlock().fit(tiny_cves.drop(columns=[schema.CVSS_ATTACK_VECTOR]))
