"""Custom strategies for hypothesis testing. """
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

from hypothesis import assume


# array comparison helpers robust to precision loss
def assert_eq(x, y, atol=np.finfo(np.float64).eps, rtol=1e-7):
    """Robustly and symmetrically assert x == y componentwise."""
    tol = atol + rtol * np.maximum(np.abs(x), np.abs(y), dtype=np.float64)
    np.testing.assert_array_less(np.abs(x - y), tol)


def assert_leq(x, y, atol=np.finfo(np.float64).eps, rtol=1e-7):
    """Robustly assert x <= y componentwise."""
    mask = np.greater(x, y)
    np.testing.assert_allclose(x[mask], y[mask], atol=atol, rtol=rtol)


# data generation strategies
def clean_floats(min_value=-1e30, max_value=1e30, width=64):
    """Custom floating point number strategy.

    Summing very large or very small floats (e.g. for means) leads to overflow
    problems. To avoid this we assume ``reasonable`` numbers for our tests.
    We exclude NaN, infinity, and negative infinity.

    The following ranges are recommended, so that squares (e.g. for variances)
    stay within the data type limits:
    -1e30 to +1e30 for 64-bit floats. (default)
    -1e15  to +1e15  for 32-bit floats.
    -200   to +200   for 16-bit floats.

    If your code really runs into floats outside this range probably something
    is wrong somewhere else.
    """
    if width == 64:
        min_value, max_value = np.clip(
            (min_value, max_value), -1e30, 1e30
        ).astype(np.float64)
    elif width == 32:
        min_value, max_value = np.clip(
            (min_value, max_value), -1e15, 1e15
        ).astype(np.float32)
    elif width == 16:
        min_value, max_value = np.clip(
            (min_value, max_value), -200, 200
        ).astype(np.float16)
    else:
        raise ValueError(
            "Invalid width parameted, expected 16, 32, or 64"
            "but got {}.".format(width)
        )
    return st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=min_value,
        max_value=max_value,
        width=width,
    )


@st.composite
def batched_float_array_iterator(
    draw, min_batch_size=None, min_data_size=None
):
    """Float array iterator strategy.

    The iterator yields batched arrays of arbitrary but fixed shape with at
    least two dimensions (the first for batch_size, the remaining for the true
    data dimensions) of any floating point data type.

    A minimum size for the batch dimension and the product of all remaining
    dimension can be specified respectively.
    """
    dtype = draw(hnp.floating_dtypes())
    bytes = dtype.itemsize
    bits = 8 * bytes
    if min_batch_size and min_data_size:
        side = max(min_batch_size, min_data_size)
    else:
        side = 1
    shape = draw(hnp.array_shapes(min_dims=2, min_side=side))
    if min_batch_size is not None:
        assume(shape[0] >= min_batch_size)
    if min_data_size is not None:
        assume(np.prod(shape[1:]) >= min_data_size)
    ar = hnp.arrays(
        dtype,
        shape,
        elements=clean_floats(width=bits),
        fill=clean_floats(width=bits),
    )
    iter = draw(st.iterables(ar, min_size=1, max_size=10))
    return iter


@st.composite
def batched_float_array_tuple_iterator(
    draw, min_batch_size=None, min_data_size=None
):
    """Float array tuple iterator strategy.

    The iterator yields tuples of batched arrays of arbitrary but fixed shape
    with at least two dimensions (the first for batch_size, the remaining for
    the true data dimensions) of any floating point data type.

    A minimum size for the batch dimension and the product of all remaining
    dimension can be specified respectively.
    """
    dtype = draw(hnp.floating_dtypes())
    bytes = dtype.itemsize
    bits = 8 * bytes
    if min_batch_size and min_data_size:
        side = max(min_batch_size, min_data_size)
    else:
        side = 1
    shape = draw(hnp.array_shapes(min_dims=2, min_side=side))
    if min_batch_size is not None:
        assume(shape[0] >= min_batch_size)
    if min_data_size is not None:
        assume(np.prod(shape[1:]) >= min_data_size)
    ar = hnp.arrays(
        dtype,
        shape,
        elements=clean_floats(width=bits),
        fill=clean_floats(width=bits),
    )
    iter = draw(st.iterables(st.tuples(ar, ar), min_size=1, max_size=10))
    return iter


@st.composite
def batched_int_array_iterator(draw, min_batch_size=None, min_data_size=None):
    """Integer array iterator strategy.

    The iterator yields batched arrays of arbitrary but fixed shape with at
    least two dimensions (the first for batch_size, the remaining for the true
    data dimensions) of any integer data type.

    A minimum size for the batch dimension and the product of all remaining
    dimension can be specified respectively.
    """
    dtype = draw(hnp.integer_dtypes())
    if min_batch_size and min_data_size:
        side = max(min_batch_size, min_data_size)
    else:
        side = 1
    shape = draw(hnp.array_shapes(min_dims=2, min_side=side))
    if min_batch_size is not None:
        assume(shape[0] >= min_batch_size)
    if min_data_size is not None:
        assume(np.prod(shape[1:]) >= min_data_size)
    ar = hnp.arrays(dtype, shape)
    iter = draw(st.iterables(ar, min_size=1, max_size=10))
    return iter


@st.composite
def batched_int_array_tuple_iterator(
    draw, min_batch_size=None, min_data_size=None
):
    """Integer array tuple iterator strategy.

    The iterator yields tuples of batched arrays of arbitrary but fixed shape
    with at least two dimensions (the first for batch_size, the remaining for
    the true data dimensions) of any integer data type.

    A minimum size for the batch dimension and the product of all remaining
    dimension can be specified respectively.
    """
    dtype = draw(hnp.integer_dtypes())
    if min_batch_size and min_data_size:
        side = max(min_batch_size, min_data_size)
    else:
        side = 1
    shape = draw(hnp.array_shapes(min_dims=2, min_side=side))
    if min_batch_size is not None:
        assume(shape[0] >= min_batch_size)
    if min_data_size is not None:
        assume(np.prod(shape[1:]) >= min_data_size)
    ar = hnp.arrays(dtype, shape)
    iter = draw(st.iterables(st.tuples(ar, ar), min_size=1, max_size=10))
    return iter
