"""Tests for `statstream.approximate`. """
import hypothesis.strategies as st
import numpy as np

from hypothesis import given

import statstream.approximate
import statstream.exact

from .strategies import assert_eq  # assert_leq,
from .strategies import (
    batched_float_array_iterator,
    batched_float_array_tuple_iterator,
    batched_int_array_iterator,
    batched_int_array_tuple_iterator,
)


# helper class for iterators (from lists) with reset function
class ResettableIterator(object):
    def __init__(self, L):
        self.list = L
        self.iter = iter(L)

    def __iter__(self):
        return self.iter

    def __next__(self):
        return next(self.iter)

    next = __next__  # for Python 2 compatibility

    def reset(self):
        self.iter = iter(self.list)


# consistency tests
@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2),
    st.integers(min_value=1, max_value=10) | st.none(),
    st.booleans(),
    st.booleans(),
)
def test_mean_and_low_rank_cov_eq_mean_and_low_rank_cov(
    X, rank, tree, reset_func
):
    """Combined mean and low rank covariance matrix is equal to individual mean
    and low rank covariance matrix."""
    # setup reset functionality for reusable iterator
    X_list = list(X)
    if reset_func:

        def reset(x):
            return iter(X_list)

        X1, X2 = iter(X_list), iter(X_list)
    else:
        reset = None
        X1, X2 = ResettableIterator(X_list), ResettableIterator(X_list)
    mean1 = statstream.exact.streaming_mean(iter(X_list))
    cov1 = statstream.approximate.streaming_low_rank_cov(
        X1,
        rank,
        tree=tree,
        reset=reset,
    )
    mean2, cov2 = statstream.approximate.streaming_mean_and_low_rank_cov(
        X2,
        rank,
        tree=tree,
        reset=reset,
    )
    # we can not directly compare results yet, factors are not uniquely defined
    cov1 = np.reshape(cov1, [cov1.shape[0], np.prod(cov1.shape[1:])])
    cov2 = np.reshape(cov2, [cov2.shape[0], np.prod(cov2.shape[1:])])
    # scipy svd used for the low rank approximations is non deterministic
    # and even the same call twice will yield different results
    # errors depend on the conditioning of the matrices
    # thus we have to be quite genererous with the tolerance here
    assert_eq(mean1, mean2)
    assert_eq(
        np.matmul(cov1.T, cov1),
        np.matmul(cov2.T, cov2),
        atol=1e-3,
        rtol=3e0,
    )


# comparison tests for using ``steps`` argument or not
@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2),
    st.integers(min_value=1, max_value=10) | st.none(),
    st.booleans(),
)
def test_autocorrelation_step_eq_no_step(X, rank, tree):
    """Low rank autocorrelation matrix is equal to low rank autocorrelation
    matrix with ``steps``."""
    X_list = list(X)  # setup reusable iterator
    corr1 = statstream.approximate.streaming_low_rank_autocorrelation(
        iter(X_list),
        rank,
        tree=tree,
    )
    corr2 = statstream.approximate.streaming_low_rank_autocorrelation(
        iter(X_list),
        rank,
        steps=len(X_list),
        tree=tree,
    )
    # we can not directly compare results yet, factors are not uniquely defined
    corr1 = np.reshape(corr1, [corr1.shape[0], np.prod(corr1.shape[1:])])
    corr2 = np.reshape(corr2, [corr2.shape[0], np.prod(corr2.shape[1:])])
    # scipy svd used for the low rank approximations is non deterministic
    # and even the same call twice will yield different results
    # errors depend on the conditioning of the matrices
    # thus we have to be quite genererous with the tolerance here
    assert_eq(
        np.matmul(corr1.T, corr1),
        np.matmul(corr2.T, corr2),
        atol=1e-3,
        rtol=3e0,
    )


@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2),
    st.integers(min_value=1, max_value=10) | st.none(),
    st.booleans(),
    st.booleans(),
)
def test_cov_step_eq_no_step(X, rank, tree, reset_func):
    """Low rank covariance matrix is equal to low rank covariance matrix with
    ``steps``."""
    X_list = list(X)
    if reset_func:

        def reset(x):
            return iter(X_list)

        X1, X2 = iter(X_list), iter(X_list)
    else:
        reset = None
        X1, X2 = ResettableIterator(X_list), ResettableIterator(X_list)
    cov1 = statstream.approximate.streaming_low_rank_cov(
        X1,
        rank,
        tree=tree,
        reset=reset,
    )
    cov2 = statstream.approximate.streaming_low_rank_cov(
        X2,
        rank,
        steps=len(X_list),
        tree=tree,
        reset=reset,
    )
    # we can not directly compare results yet, factors are not uniquely defined
    cov1 = np.reshape(cov1, [cov1.shape[0], np.prod(cov1.shape[1:])])
    cov2 = np.reshape(cov2, [cov2.shape[0], np.prod(cov2.shape[1:])])
    # scipy svd used for the low rank approximations is non deterministic
    # and even the same call twice will yield different results
    # errors depend on the conditioning of the matrices
    # thus we have to be quite genererous with the tolerance here
    assert_eq(
        np.matmul(cov1.T, cov1),
        np.matmul(cov2.T, cov2),
        atol=1e-3,
        rtol=3e0,
    )
