"""Tests for `statstream.exact`. """
from itertools import tee

import numpy as np

from hypothesis import given
from scipy.linalg import cholesky

import statstream.exact

from .strategies import (
    assert_eq,
    assert_leq,
    batched_float_array_iterator,
    batched_float_array_tuple_iterator,
    batched_int_array_iterator,
    batched_int_array_tuple_iterator,
)


# comparison tests
@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_min_leq_max(X):
    """Minimum is less or equal to maximum."""
    X1, X2 = tee(X)  # make buffered copy of iterator to use it twice
    min = statstream.exact.streaming_min(X1)
    max = statstream.exact.streaming_max(X2)
    assert_leq(min, max)


@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_min_leq_mean(X):
    """Minimum is less or equal to mean."""
    X1, X2 = tee(X)  # make buffered copy of iterator to use it twice
    min = statstream.exact.streaming_min(X1)
    mean = statstream.exact.streaming_mean(X2)
    assert_leq(min, mean)


@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_mean_leq_max(X):
    """Mean is less or equal to maximum."""
    X1, X2 = tee(X)  # make buffered copy of iterator to use it twice
    mean = statstream.exact.streaming_mean(X1)
    max = statstream.exact.streaming_max(X2)
    assert_leq(mean, max)


@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_var_geq_zero(X):
    """Variance is greater or equal to zero."""
    var = statstream.exact.streaming_var(X)
    assert np.alltrue(np.greater_equal(var, 0.0))


@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_std_geq_zero(X):
    """Standard deviation is greater or equal to zero."""
    std = statstream.exact.streaming_std(X)
    assert np.alltrue(np.greater_equal(std, 0.0))


# consistency tests
@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_mean_and_var_eq_mean_and_var(X):
    """Combined mean and variance is equal to individual mean and variance."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    mean1 = statstream.exact.streaming_mean(X1)
    var1 = statstream.exact.streaming_var(X2)
    mean2, var2 = statstream.exact.streaming_mean_and_var(X3)
    assert_eq(mean1, mean2)
    assert_eq(var1, var2, atol=np.sqrt(np.finfo(np.float64).eps))


@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_mean_and_std_eq_mean_and_std(X):
    """Combined mean and standard deviation is equal to individual mean and
    standard deviation.
    """
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    mean1 = statstream.exact.streaming_mean(X1)
    std1 = statstream.exact.streaming_std(X2)
    mean2, std2 = statstream.exact.streaming_mean_and_std(X3)
    assert_eq(mean1, mean2)
    assert_eq(std1, std2, atol=np.sqrt(np.finfo(np.float64).eps))


@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2)
)
def test_mean_and_cov_eq_mean_and_cov(X):
    """Combined mean and covariance matrix is equal to individual mean and
    covariance matrix.
    """
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    mean1 = statstream.exact.streaming_mean(X1)
    cov1 = statstream.exact.streaming_cov(X2)
    mean2, cov2 = statstream.exact.streaming_mean_and_cov(X3)
    assert_eq(mean1, mean2)
    assert_eq(cov1, cov2, atol=np.sqrt(np.finfo(np.float64).eps))


@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2)
)
def test_diag_cov_eq_var(X):
    """Diagonal of covariance matrix is equal to variance."""
    X1, X2 = tee(X)  # make buffered copy of iterator to use it twice
    var = statstream.exact.streaming_var(X1)
    cov = statstream.exact.streaming_cov(X2)
    shape, ndim = cov.shape, cov.ndim
    assert ndim % 2 == 0
    assert np.array_equal(shape[: ndim // 2], shape[ndim // 2 :])
    prod = np.prod(shape[: ndim // 2])
    cov_matrix = np.reshape(cov, (prod, prod))
    assert_eq(cov_matrix, cov_matrix.T)
    diag = np.diag(cov_matrix)
    assert_eq(
        var,
        np.reshape(diag, shape[: ndim // 2]),
        atol=np.sqrt(np.finfo(np.float64).eps),
    )


# property tests
@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2)
)
def test_cov_is_symmetric(X):
    """Covariance matrix is square and symmetric."""
    cov = statstream.exact.streaming_cov(X)
    shape, ndim = cov.shape, cov.ndim
    assert ndim % 2 == 0
    assert np.array_equal(shape[: ndim // 2], shape[ndim // 2 :])
    covT = np.transpose(
        cov,
        axes=np.concatenate(
            (np.arange(ndim // 2, ndim), np.arange(0, ndim // 2))
        ),
    )
    assert_eq(cov, covT)


@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2)
)
def test_cov_is_psd(X):
    """Covariance matrix is symmetric positive semidefinite.

    In the positive definite case this is equivalent to the existance of a
    Cholesky decomposition. To handle also semidefinite cases we add a small
    identity shift to the diagonal.
    """
    cov = statstream.exact.streaming_cov(X)
    shape, ndim = cov.shape, cov.ndim
    assert ndim % 2 == 0
    assert np.array_equal(shape[: ndim // 2], shape[ndim // 2 :])
    prod = np.prod(shape[: ndim // 2])
    cov_matrix = np.reshape(cov, (prod, prod))
    assert_eq(cov_matrix, cov_matrix.T)
    # cholesky decomposition raises LinAlgError if not positive definite
    rescale = max(1.0, np.abs(cov_matrix).max())
    cov_matrix /= rescale
    shift = np.sqrt(np.finfo(np.float64).eps)
    cov_matrix[np.diag_indices_from(cov_matrix)] += shift
    cholesky(cov_matrix)


# comparison tests for using ``steps`` argument or not
@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_min_step_eq_no_step(X):
    """Minimum is equal to minimum with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    min1 = statstream.exact.streaming_min(X1)
    X_list = list(X2)
    min2 = statstream.exact.streaming_min(X3, steps=len(X_list))
    assert_eq(min1, min2)


@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_max_step_eq_no_step(X):
    """Maximum is equal to maximum with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    max1 = statstream.exact.streaming_max(X1)
    X_list = list(X2)
    max2 = statstream.exact.streaming_max(X3, steps=len(X_list))
    assert_eq(max1, max2)


@given(
    batched_float_array_iterator()
    | batched_int_array_iterator()
    | batched_float_array_tuple_iterator()
    | batched_int_array_tuple_iterator()
)
def test_mean_step_eq_no_step(X):
    """Mean is equal to mean with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    mean1 = statstream.exact.streaming_mean(X1)
    X_list = list(X2)
    mean2 = statstream.exact.streaming_mean(X3, steps=len(X_list))
    assert_eq(mean1, mean2)


@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_var_step_eq_no_step(X):
    """Variance is equal to variance with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    var1 = statstream.exact.streaming_var(X1)
    X_list = list(X2)
    var2 = statstream.exact.streaming_var(X3, steps=len(X_list))
    assert_eq(var1, var2)


@given(
    batched_float_array_iterator(min_batch_size=2)
    | batched_int_array_iterator(min_batch_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2)
)
def test_std_step_eq_no_step(X):
    """Standard deviation is equal to standard deviation with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    std1 = statstream.exact.streaming_std(X1)
    X_list = list(X2)
    std2 = statstream.exact.streaming_std(X3, steps=len(X_list))
    assert_eq(std1, std2)


@given(
    batched_float_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_iterator(min_batch_size=2, min_data_size=2)
    | batched_float_array_tuple_iterator(min_batch_size=2, min_data_size=2)
    | batched_int_array_tuple_iterator(min_batch_size=2, min_data_size=2)
)
def test_cov_step_eq_no_step(X):
    """Covariance matrix is equal to covariance matrix with ``steps``."""
    X1, X2, X3 = tee(X, 3)  # make buffered copy of iterator to use it thrice
    cov1 = statstream.exact.streaming_cov(X1)
    X_list = list(X2)
    cov2 = statstream.exact.streaming_cov(X3, steps=len(X_list))
    assert_eq(cov1, cov2)
