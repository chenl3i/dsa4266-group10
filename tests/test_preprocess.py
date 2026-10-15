import numpy as np
import pandas as pd
import pytest

from ugr16ad.preprocess import Preprocessor

FIT = pd.DataFrame({"a": [0, 1, 3, 7], "b": [2, 2, 2, 2], "c": [0, 0, 0, 100]})


def test_fit_drops_constant_columns_and_standardises():
    pre = Preprocessor().fit(FIT)
    assert pre.feature_names_ == ["a", "c"]
    Z = pre.transform(FIT)
    assert Z.dtype == np.float32
    assert Z.shape == (4, 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-6)


def test_transform_reuses_fit_statistics_unchanged():
    pre = Preprocessor().fit(FIT)
    later = pd.DataFrame({"a": [0], "b": [9], "c": [0]})
    logged = np.log1p(FIT[["a", "c"]].to_numpy(dtype=float))
    expected = (np.log1p(0) - logged.mean(axis=0)) / logged.std(axis=0)
    Z = pre.transform(later)
    # "b" is constant in the fit part, so it stays dropped even though it varies later.
    assert Z.shape == (1, 2)
    np.testing.assert_allclose(Z[0], expected, rtol=1e-5)


def test_transform_follows_feature_names_order_not_input_order():
    pre = Preprocessor().fit(FIT)
    shuffled = FIT[["c", "b", "a"]]
    np.testing.assert_array_equal(pre.transform(shuffled), pre.transform(FIT))


def test_transform_ignores_extra_columns():
    pre = Preprocessor().fit(FIT)
    extra = FIT.assign(z=[5, 6, 7, 8])
    np.testing.assert_array_equal(pre.transform(extra), pre.transform(FIT))


def test_transform_does_not_need_columns_dropped_at_fit():
    pre = Preprocessor().fit(FIT)
    # "b" was constant at fit, so it is not in feature_names_ and may be absent.
    np.testing.assert_array_equal(pre.transform(FIT.drop(columns="b")), pre.transform(FIT))


def test_transform_before_fit_raises():
    pre = Preprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(FIT)
    assert pre.fit(FIT).transform(FIT).shape == (4, 2)


def test_transform_rejects_different_columns():
    pre = Preprocessor().fit(FIT)
    with pytest.raises(ValueError):
        pre.transform(FIT.drop(columns="a"))


def test_transform_rejects_missing_column_even_with_extra_columns_present():
    pre = Preprocessor().fit(FIT)
    # Same number of columns as at fit time, but "a" is gone.
    swapped = FIT.drop(columns="a").assign(z=[1, 2, 3, 4])
    with pytest.raises(ValueError):
        pre.transform(swapped)
