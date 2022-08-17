"""Exact statistics for streaming data.

The `statstream.exact` module provides functions for statistics that
can be exactly computed from streamed data.

This includes for example mean, variance, minimum, and maximum.

"""
import numpy as np

from tqdm import tqdm


def streaming_min(X, steps=None):
    """Minimum of a streaming dataset.

    Computes the minimum of a dataset from a stream of batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the minimum
    calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The minimum has the same shape as
    the remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]``
    will produce a componentwise minimum of shape ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The componentwise minimum of the seen data samples.

    See Also
    --------
    streaming_max: get the componentwise maximum in a single pass.
    """

    def _process_batch(batch, min):
        batch_min = np.min(batch, axis=0)
        return np.minimum(batch_min, min)

    min = np.inf
    if steps:
        for step in tqdm(range(steps), "min calculation"):
            batch = next(X)
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            min = _process_batch(batch, min)
    else:
        for batch in tqdm(X, "min calculation"):
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            min = _process_batch(batch, min)
    return min


def streaming_max(X, steps=None):
    """Maximum of a streaming dataset.

    Computes the maximum of a dataset from a stream of batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the maximum
    calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The maximum has the same shape as
    the remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]``
    will produce a componentwise maximum of shape ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The componentwise maximum of the seen data samples.

    See Also
    --------
    streaming_min: get the componentwise minimum in a single pass.
    """

    def _process_batch(batch, max):
        batch_max = np.max(batch, axis=0)
        return np.maximum(batch_max, max)

    max = -np.inf
    if steps:
        for step in tqdm(range(steps), "max calculation"):
            batch = next(X)
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            max = _process_batch(batch, max)
    else:
        for batch in tqdm(X, "max calculation"):
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            max = _process_batch(batch, max)
    return max


def streaming_mean(X, steps=None):
    """Mean of a streaming dataset.

    Computes the mean of a dataset from a stream of batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the mean
    calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The mean has the same shape as
    the remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]``
    will produce a mean of shape ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The mean of the seen data samples.

    See Also
    --------
    streaming_mean_and_var: get both mean and variance in a single pass.
    """

    def _process_batch(batch, accumulator, count):
        batch_size = batch.shape[0]
        if accumulator is not None:
            batch_extended = np.concatenate(
                [np.expand_dims(accumulator, 0), batch], axis=0
            )
            accumulator = np.sum(batch_extended, axis=0, dtype=np.float64)
        else:
            accumulator = np.sum(batch, axis=0, dtype=np.float64)
        count += batch_size
        return accumulator, count

    accumulator, count = None, 0
    if steps:
        for step in tqdm(range(steps), "mean calculation"):
            batch = next(X)
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            accumulator, count = _process_batch(batch, accumulator, count)
    else:
        for batch in tqdm(X, "mean calculation"):
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            accumulator, count = _process_batch(batch, accumulator, count)
    return accumulator / count


def streaming_var(X, steps=None):
    """Variance of a streaming dataset.

    Computes the variance of a dataset from a stream of batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the variance
    calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The variance has the same shape as
    the remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]``
    will produce variance of shape ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The variance of the seen data samples.

    See Also
    --------
    streaming_mean_and_var: get the mean and variance in a single pass.

    Notes
    -----
    Computing variances necessarily includes computing the mean as well,
    so there is no computational benefit of using `streaming_var` over
    using `streaming_mean_and_var`.

    The streamed variances are calculated as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """
    mean, variance = streaming_mean_and_var(X, steps)
    return variance


def streaming_mean_and_var(X, steps=None):
    """Mean and variance of a streaming dataset.

    Computes the mean and variance of a dataset from a stream of batches of
    samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the mean
    and variance calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The mean and variance have the same
    shape as the remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` will produce a mean and variance of shape
    ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The mean of the seen data samples.
    array
        The variance of the seen data samples.

    See Also
    --------
    streaming_mean: get only the mean in a single pass.
    streaming_var: get only the variance in a single pass.

    Notes
    -----
    The streamed variances are calculated as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """

    def _process_batch(batch, mean_accumulator, var_accumulator, count):
        batch_size = batch.shape[0]
        if mean_accumulator is not None:
            batch_sum = np.sum(batch, axis=0, dtype=np.float64)
            diff = mean_accumulator / count - batch_sum / batch_size
            correction = (
                np.square(diff) / (count + batch_size) * count * batch_size
            )
            batch_extended = np.concatenate(
                [np.expand_dims(mean_accumulator, 0), batch], axis=0
            )
            mean_accumulator = np.sum(batch_extended, axis=0, dtype=np.float64)
            batch_var = np.var(
                batch, axis=0, ddof=batch_size - 1, dtype=np.float64
            )
            var_accumulator += batch_var + correction
        else:
            mean_accumulator = np.sum(batch, axis=0, dtype=np.float64)
            var_accumulator = np.var(
                batch, axis=0, ddof=batch_size - 1, dtype=np.float64
            )
        count += batch_size
        return mean_accumulator, var_accumulator, count

    mean_accumulator, variance_accumulator, count = None, None, 0
    if steps:
        for step in tqdm(range(steps), "variance calculation"):
            batch = next(X)
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            mean_accumulator, variance_accumulator, count = _process_batch(
                batch, mean_accumulator, variance_accumulator, count
            )
    else:
        for batch in tqdm(X, "variance calculation"):
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            mean_accumulator, variance_accumulator, count = _process_batch(
                batch, mean_accumulator, variance_accumulator, count
            )
    return mean_accumulator / count, variance_accumulator / (count - 1)


def streaming_std(X, steps=None):
    """Standard deviation of a streaming dataset.

    Computes the standard deviation of a dataset from a stream of batches of
    samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the standard
    deviation calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The standard deviation has the same
    shape as the remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` will produce a mean and variance of shape
    ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The standard deviation of the seen data samples.

    See Also
    --------
    streaming_mean_and_std: get mean and standard deviation in a single pass.

    Notes
    -----
    Computing standard deviation necessarily includes computing the mean,
    so there is no computational benefit of using `streaming_std` over
    using `streaming_mean_and_std`. This function does nothing else than
    taking the square root of `streaming_var`.

    The streamed variances are calculated as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """
    return np.sqrt(streaming_var(X, steps))


def streaming_mean_and_std(X, steps=None):
    """Mean and standard deviation of a streaming dataset.

    Computes the mean and the standard deviation of a dataset from a stream of
    batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the mean and
    standard deviation calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The mean and standard deviation
    have the same shape as the remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` will produce a mean and standard deviation of
    shape ``[d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The mean of the seen data samples.
    array
        The standard deviation of the seen data samples.

    See Also
    --------
    streaming_mean: get only the mean in a single pass.
    streaming_std: get only the standard deviation in a single pass.

    Notes
    -----
    This function does nothing else than `streaming_mean_and_var` followed
    by taking the square root of the variance.

    The streamed variances are calculated as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """
    mean, variance = streaming_mean_and_var(X, steps)
    return mean, np.sqrt(variance)


def streaming_cov(X, steps=None):
    """Covariance matrix of a streaming dataset.

    Computes the the covariance matrix of a dataset from a stream of
    batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the
    covariance calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The covariance has the squared
    shape of as remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` will produce a covariance of shape
    ``[d1, ..., dN, d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The covariance matrix of the seen data samples.

    Warnings
    --------
    Use this function only on data sets of reasonably small dimensions.

    Full covariances matrices are costly to compute and require large amounts
    of memory. The shape of the covariance matrix is squared the size of each
    individual sample in the data set.

    If your data is high dimensional and you do not need the exact covariance
    matrix, consider using `streaming_mean_and_low_rank_cov` or
    `streaming_low_rank_cov` from `statstream.approximate` instead.

    See Also
    --------
    streaming_mean_and_cov: get mean and covariance in a single pass.

    Notes
    -----
    Computing covariances necessarily includes computing the mean,
    so there is no computational benefit of using `streaming_cov` over
    using `streaming_mean_and_cov`.

    The streamed covariance calculation is a generalization of the streamed
    variance calculation as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """
    mean, covariance = streaming_mean_and_cov(X, steps)
    return covariance


def streaming_mean_and_cov(X, steps=None):
    """Mean and covariance matrix of a streaming dataset.

    Computes the mean and the covariance matrix of a dataset from a stream of
    batches of samples.
    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the mean and
    covariance calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The mean has the same shape as the
    remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]`` will
    produce a mean of shape ``[d1, ..., dN]``. Covariances are arranged in the
    squared shape ``[d1, ..., dN, d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding batches of samples.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.

    Returns
    -------
    array
        The mean of the seen data samples.
    array
        The covariance matrix of the seen data samples.

    Warnings
    --------
    Use this function only on data sets of reasonably small dimensions.

    Full covariances matrices are costly to compute and require large amounts
    of memory. The shape of the covariance matrix is squared the size of each
    individual sample in the data set.

    If your data is high dimensional and you do not need the exact covariance
    matrix, consider using `streaming_mean_and_low_rank_cov` or
    `streaming_low_rank_cov` from `statstream.approximate` instead.

    See Also
    --------
    streaming_mean: get only the mean in a single pass.
    streaming_cov: get only the covariance matrix in a single pass.

    Notes
    -----
    The streamed covariance calculation is a generalization of the streamed
    variance calculation as described in [1]_.

    References
    ----------
    .. [1] Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
           "Updating formulae and a pairwise algorithm for computing sample
           variances", 1979.
    """

    def _process_batch(batch, mean_accumulator, cov_accumulator, count):
        batch_size = batch.shape[0]
        if mean_accumulator is not None:
            batch_sum = np.sum(batch, axis=0, dtype=np.float64)
            diff = np.reshape(
                mean_accumulator / count - batch_sum / batch_size,
                [np.prod(batch_sum.shape), 1],
            )
            diff_prod = np.dot(diff, diff.T)
            correction = diff_prod / (count + batch_size) * count * batch_size
            batch_extended = np.concatenate(
                [np.expand_dims(mean_accumulator, 0), batch], axis=0
            )
            mean_accumulator = np.sum(batch_extended, axis=0, dtype=np.float64)
            batch_cov = np.cov(
                np.reshape(batch, [batch_size, -1]),
                rowvar=False,
                ddof=batch_size - 1,
            )
            cov_accumulator += batch_cov + correction
        else:
            mean_accumulator = np.sum(batch, axis=0, dtype=np.float64)
            cov_accumulator = np.cov(
                np.reshape(batch, [batch_size, -1]),
                rowvar=False,
                ddof=batch_size - 1,
            )
        count += batch_size
        return mean_accumulator, cov_accumulator, count

    mean_accumulator, covariance_accumulator, count = None, None, 0
    if steps:
        for step in tqdm(range(steps), "mean and covariance calculation"):
            batch = next(X)
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            mean_accumulator, covariance_accumulator, count = _process_batch(
                batch, mean_accumulator, covariance_accumulator, count
            )
    else:
        for batch in tqdm(X, "mean and covariance calculation"):
            if isinstance(batch, tuple) and len(batch) > 1:
                batch = batch[0]
            mean_accumulator, covariance_accumulator, count = _process_batch(
                batch, mean_accumulator, covariance_accumulator, count
            )
    covariance_accumulator = np.reshape(
        covariance_accumulator,
        2 * mean_accumulator.shape,
    )
    return mean_accumulator / count, covariance_accumulator / (count - 1)


# aliases
streaming_variance = streaming_var
streaming_mean_and_variance = streaming_mean_and_var
streaming_covariance = streaming_cov
streaming_mean_and_covariance = streaming_mean_and_cov
s_min = streaming_min
s_max = streaming_max
s_mean = streaming_mean
s_var = streaming_var
s_variance = streaming_variance
s_std = streaming_std
s_cov = streaming_cov
s_covariance = streaming_covariance
s_mean_and_var = streaming_mean_and_var
s_mean_and_variance = streaming_mean_and_variance
s_mean_and_std = streaming_mean_and_std
s_mean_and_cov = streaming_mean_and_cov
s_mean_and_covariance = streaming_mean_and_covariance
