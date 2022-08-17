"""Approximate statistics for streaming data.

The `statstream.approximate` module provides functions for statistics that can
not be exactly computed from streamed data.

This includes for example low rank factorisations of covariance matrices.

"""
import numpy as np

from scipy.linalg import svd
from scipy.sparse.linalg import svds
from tqdm import tqdm

from .exact import streaming_mean


def _truncated_svd(A, rank=None):
    """Private helper function for calculating truncated singular value
    decompositions.

    Given a matrix A and a rank K computes the truncated singular value
    decomposition U, S, V of A where the K largest singular values are kept and
    the rest are truncated to zero. If A has shape [N, M] then U, S, V have
    shapes [N, K], [K], [M, K] respectively, and U @ np.diag(S) @ V.T is the
    best rank K approximation to A with respect to the spectral norm.

    This function internally uses standard numpy and scipy routines for
    calculating the SVD, it merely serves as a `syntactic sugar` container
    for deciding which subroutines to use depending on the shape of A and
    the rank K.

    Parameters
    ----------
    A : array
        The matrix to decompose.
    rank : int
        Number of singular values to keep for the truncated SVD.

    Returns
    -------
    S : array
        (Truncated) singular values of A.
    U, V : array
        matrix factors of the (truncated) singular value decomposition of A.
    """
    if not rank:
        rank = np.min(A.shape)
    if rank < np.min(A.shape):
        # use truncated SVD if rank is reduced
        U, S, VT = svds(A.astype(np.float64), rank)
    else:
        # use full SVD otherwise
        U, S, VT = svd(A.astype(np.float64), full_matrices=False)
    V = VT.T
    return U, S, V


def _merge_low_rank_eigendecomposition(S1, V1, S2, V2, rank=None):
    """Private helper function for merging SVD based low rank approximations.

    Given factors S1, V1 and S2, V2 of shapes [K1], [M, K1] and [K2], [M, K2]
    respectively of singular value decompositions

        A1 = U1 @ np.diag(S1) @ V1.T
        A2 = U2 @ np.diag(S2) @ V2.T

    merge them into factors S, V of shape [K], [M, K] of an approximate
    decomposition A = U @ np.diag(S) @ V.T, where A is the concatenation of A1
    and A2 along the first axis. This is done without the need of calculating
    U1, U2, and U.
    This is useful for merging eigendecompositions V @ np.diag(S**2) @ V.T of
    autocorrelation (or similarly covariance) matrices A.T @ A that do not
    require U. Using truncated singular value decompositons can be used for
    merging low rank approximations.

    Parameters
    ----------
    S1 : array
        Singular values of first matrix.
    V1 : array
        Factor of the singular value decomposition of first matrix.
    S2 : array
        Singular values of second matrix.
    V2 : array
        Factor of the singular value decomposition of second matrix.
    rank : int
        Number of singular values to keep after merging. If set to `None`
        no truncation will be done, thus rank will equal the sum of
        singular values given in S1 and S2.

    Returns
    -------
    S : array
        (Truncated) singular values of the singular value decomposition of
        concatenated matrix.
    V : array
        Factor of the singular value decomposition of concatenated matrix.

    Notes
    -----
    The algorithm for combining SVD based low rank approximations is
    described in more detail in [1]_.

    References
    ----------
    .. [1] Radim, Rehurek,
           "Scalability of Semantic Analysis in Natural Language Processing",
           2011.
    """
    rank1, rank2 = S1.size, S2.size
    if not rank or rank > rank1 + rank2:
        rank = rank1 + rank2
    if rank > min(V1.shape[0], V2.shape[0]):
        rank = min(V1.shape[0], V2.shape[0])
    Z = np.matmul(V1.T, V2)
    Q, R = np.linalg.qr(V2 - np.matmul(V1, Z), mode="reduced")
    Zfill = np.zeros([rank2, rank1])
    B = np.concatenate(
        [
            np.concatenate([np.diag(S1), np.matmul(Z, np.diag(S2))], axis=1),
            np.concatenate([Zfill, np.matmul(R, np.diag(S2))], axis=1),
        ],
        axis=0,
    )
    U, S, VT = _truncated_svd(B, rank=rank)
    V = np.matmul(V1, U[:rank1, :]) + np.matmul(Q, U[rank1:, :])
    return S, V


def streaming_low_rank_autocorrelation(
    X, rank, steps=None, shift=0.0, tree=False
):
    """Low rank factorization of the sample autocorrelation matrix of a
    streaming dataset.

    Computes a factorization of the autocorrelation matrix of a dataset from
    a stream of  batches of samples. If the full data set was given in a matrix
    ``A`` of shape ``[N, M]``, where ``N`` is the number of data samples and
    ``M`` is the dimensionality of each sample, then the autocorrelation matrix
    is ``1/(N-1)*A.T @ A`` and of shape ``[M, M]``.
    The function computes a matrix ``L`` of shape ``[M, K]`` such that
    ``L.T @ L`` is an approximation of the autocorrelation matrix of rank at
    most ``K``.

    This is done from a stream of sample batches without ever explicitly
    forming matrices of the full shape ``[M, M]``. Batches can be combined in a
    *linear* streaming way (which gives more relative weight to later batches)
    or in *binary tree* mode, where batches are combined pairwise, then the
    results are combined again pairwise and so on (this leads to an additional
    memory requirement of a factor of ``log(N)``).

    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the
    correlation calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The correlation has the squared
    shape as the remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` will result in a correlation factor of shape
    ``[K, d1, ..., dN]``.

    This function consumes an iterator, thus finite iterators will be empty
    after a call to this function, unless ``steps`` is set to a smaller number
    than batches in the iterator.

    Parameters
    ----------
    X : iterable
        An iterator yielding the batches of samples.
    rank : int
        The maximal rank of the approximate decomposition factor.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.
    shift : array, optional
        Apply a shift of data samples before calculating correlations, that is
        use (X-shift) instead of X (must be broadcastable to the shape of
        batches from X). The default is 0.0, that is no shift is used.
    tree : bool, optional
        Use the binary tree mode to combine batches more evenly at the cost of
        additional memory requirement. The default is `False`.

    Returns
    -------
    array
        A low-rank factor of a symmetric decomposition of the autocorrelation
        matrix of the seen data.

    Notes
    -----
    The algorithm for combining SVD based low rank approximations is
    described in more detail in [1]_.


    References
    ----------
    .. [1] Radim, Rehurek,
           "Scalability of Semantic Analysis in Natural Language Processing",
           2011.
    """

    def _process_batch(batch, S, V, rank, count):
        batch_size = batch.shape[0]
        Ub, Sb, Vb = _truncated_svd(
            np.reshape(batch, [batch_size, -1]),
            rank,
        )
        if S is None or V is None:
            S, V = Sb, Vb
        else:
            S, V = _merge_low_rank_eigendecomposition(S, V, Sb, Vb, rank=rank)
        count += batch_size
        return S, V, count

    def _tree_process_batch(batch, stack, rank, count):
        batch_size = batch.shape[0]
        Ub, Sb, Vb = _truncated_svd(
            np.reshape(batch, [batch_size, -1]),
            rank,
        )
        stack.append({"S": Sb, "V": Vb, "level": 0})
        while len(stack) >= 2 and stack[-1]["level"] == stack[-2]["level"]:
            item1, item2 = stack.pop(), stack.pop()
            S, V = _merge_low_rank_eigendecomposition(
                item1["S"], item1["V"], item2["S"], item2["V"], rank=rank
            )
            stack.append({"S": S, "V": V, "level": item1["level"] + 1})
        count += batch_size
        return stack, count

    if tree:
        stack, count = [], 0
        if steps:
            for step in tqdm(range(steps), "autocorrelation approximation"):
                batch = next(X)
                if isinstance(batch, tuple) and len(batch) > 1:
                    batch = batch[0]
                stack, count = _tree_process_batch(
                    batch - shift, stack, rank, count
                )
        else:
            for batch in tqdm(X, "autocorrelation approximation"):
                if isinstance(batch, tuple) and len(batch) > 1:
                    batch = batch[0]
                stack, count = _tree_process_batch(
                    batch - shift, stack, rank, count
                )
        while len(stack) >= 2:
            item1, item2 = stack.pop(), stack.pop()
            S, V = _merge_low_rank_eigendecomposition(
                item1["S"], item1["V"], item2["S"], item2["V"], rank
            )
            stack.append({"S": S, "V": V, "level": item1["level"] + 1})
        S, V = stack[0]["S"], stack[0]["V"]
    else:
        S, V, count = None, None, 0
        if steps:
            for step in tqdm(range(steps), "autocorrelation approximation"):
                batch = next(X)
                if isinstance(batch, tuple) and len(batch) > 1:
                    batch = batch[0]
                S, V, count = _process_batch(batch - shift, S, V, rank, count)
        else:
            for batch in tqdm(X, "autocorrelation approximation"):
                if isinstance(batch, tuple) and len(batch) > 1:
                    batch = batch[0]
                S, V, count = _process_batch(batch - shift, S, V, rank, count)
    factor = V * np.expand_dims(S, 0)
    return np.reshape(factor.T, (S.size,) + batch.shape[1:]) / np.sqrt(
        count - 1
    )


def streaming_low_rank_cov(X, rank, steps=None, tree=False, reset=None):
    """Low rank factorization of the covariance matrix of a streaming dataset.

    Computes a factorization of the covariance matrix of a dataset from a
    stream of  batches of samples.
    If the full data set was given in a matrix ``A`` of shape ``[N, M]``, where
    ``N`` is the number of data samples and ``M`` is the dimensionality of each
    sample, then the covariance matrix is
    ``1/(N-1)*(A-mean(A)).T @ (A-mean(A))`` and of shape ``[M, M]``.
    The function computes a matrix ``L`` of shape ``[M, K]`` such that
    ``L.T @ L`` is an approximation of the covariance matrix of rank at most
    ``K``.
    This is done in a two-pass algorithm that first computes the mean from a
    stream of batches and then the covariance using
    `streaming_low_rank_autocorrelation` shifted by the precomputed mean.

    This is done from a stream of sample batches without ever explicitly
    forming matrices of the full shape ``[M, M]``. Batches can be combined in a
    *linear* streaming way (which gives more relative weight to later batches)
    or in *binary tree* mode, where batches are combined pairwise, then the
    results are combined again pairwise and so on (this leads to an additional
    memory requirement of a factor of ``log(N)``).

    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the
    covariance calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The full covariance would have the
    squared shape as the remaining axes, e.g. batches of shape
    ``[batch_size, d1, ..., dN]`` would result in a covariance maxtrix of shape
    ``[d1, ..., dN, d1, ..., dN]``. The low-rank covariance factor ``L`` will
    have shape ``[K, d1, ..., dN]``.

    This function consumes an iterator twice, thus only finite iterators
    can be handled and the given iterator will be empty after a call to this
    function, unless ``steps`` is set to a smaller number than batches in the
    iterator. For restarting the iterator for the second pass, a reset
    function needs to be available. This can either be passed as a seperate
    argument or be part of the iterator itself. If no reset function is
    provided as argument, the iterator ``X`` is assumed to have a ``reset()``
    method, which is called after the mean computation.

    Parameters
    ----------
    X : iterable
        An iterator yielding the batches of samples.
    rank : int
            The maximal rank of the approximate decomposition factor.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.
    tree : bool, optional
        Use the binary tree mode to combine batches more evenly at the cost of
        additional memory requirement. The default is `False`.
    reset : callable or None, optional
        A function handle to reset the iterator after the first pass for the
        mean calculation. The reset function must accept the iterator as
        argument and return a resetted iterator. If set to `None` the iterator
        is assumed to have a reset method, which will then be used. The default
        is `None`.

    Returns
    -------
    array
        A low-rank factor of a symmetric decomposition of the covariance
        matrix of the seen data.

    Notes
    -----
    Computing covariances necessarily includes computing the mean,
    so there is no computational benefit of using `streaming_low_rank_cov` over
    using `streaming_mean_and_low_rank_cov`. In fact this function internally
    uses the latter and simly discards the mean.

    The algorithm for combining SVD based low rank approximations is
    described in more detail in [1]_.

    References
    ----------
    .. [1] Radim, Rehurek,
           "Scalability of Semantic Analysis in Natural Language Processing",
           2011.
    """
    mean = streaming_mean(X, steps=steps)
    if reset:
        X = reset(X)
    else:
        X.reset()
    covariance = streaming_low_rank_autocorrelation(
        X,
        rank,
        steps=steps,
        shift=mean,
        tree=tree,
    )
    return covariance


def streaming_mean_and_low_rank_cov(
    X, rank, steps=None, tree=False, reset=None
):
    """Mean and a low rank factorization of the covariance matrix of a
    streaming dataset.

    Computes the mean and a factorization of the covariance matrix of a dataset
    from a stream of  batches of samples.
    If the full data set was given in a matrix ``A`` of shape ``[N, M]``, where
    ``N`` is the number of data samples and ``M`` is the dimensionality of each
    sample, then the covariance matrix is
    ``1/(N-1)*(A-mean(A)).T @ (A-mean(A))`` and of shape ``[M, M]``.
    The function computes a matrix ``L`` of shape ``[M, K]`` such that
    ``L.T @ L`` is an approximation of the covariance matrix of rank at most
    ``K``.
    This is done in a two-pass algorithm that first computes the mean from a
    stream of batches and then the covariance using
    `streaming_low_rank_autocorrelation` shifted by the precomputed mean.

    This is done from a stream of sample batches without ever explicitly
    forming matrices of the full shape ``[M, M]``. Batches can be combined in a
    *linear* streaming way (which gives more relative weight to later batches)
    or in *binary tree* mode, where batches are combined pairwise, then the
    results are combined again pairwise and so on (this leads to an additional
    memory requirement of a factor of ``log(N)``).

    The data has to be provided by an iterator yielding batches of samples.
    Either a number of steps can be specified, or the iterator is assumed to
    be emptied in a finite number of steps. In the first case only the given
    number of batches is extracted from the iterator and used for the mean and
    covariance calculation, even if the iterator could yield more data.

    Samples are given along the first axis. The mean has the same shape as
    the remaining axes, e.g. batches of shape ``[batch_size, d1, ..., dN]``
    will produce a mean of shape ``[d1, ..., dN]``. The covariance factor ``L``
    will have shape ``[K, d1, ..., dN]``.

    This function consumes an iterator twice, thus only finite iterators
    can be handled and the given iterator will be empty after a call to this
    function, unless ``steps`` is set to a smaller number than batches in the
    iterator. For restarting the iterator for the second pass, a reset
    function needs to be available. This can either be passed as a seperate
    argument or be part of the iterator itself. If no reset function is
    provided as argument, the iterator ``X`` is assumed to have a ``reset()``
    method, which is called after the mean computation.

    Parameters
    ----------
    X : iterable
        An iterator yielding the batches of samples.
    rank : int
            The maximal rank of the approximate decomposition factor.
    steps : int, optional
        The number of batches to use from the iterator (all available batches
        are used if set to `None`). The defaul is `None`.
    tree : bool, optional
        Use the binary tree mode to combine batches more evenly at the cost of
        additional memory requirement. The default is `False`.
    reset : callable or None, optional
        A function handle to reset the iterator after the first pass for the
        mean calculation. The reset function must accept the iterator as
        argument and return a resetted iterator. If set to `None` the iterator
        is assumed to have a reset method, which will then be used. The default
        is `None`.

    Returns
    -------
    array
        The mean of the seen data samples.
    array
        A low-rank factor of a symmetric decomposition of the covariance
        matrix of the seen data.

    Notes
    -----
    The algorithm for combining SVD based low rank approximations is
    described in more detail in [1]_.

    References
    ----------
    .. [1] Radim, Rehurek,
           "Scalability of Semantic Analysis in Natural Language Processing",
           2011.
    """
    mean = streaming_mean(X, steps=steps)
    if reset:
        X = reset(X)
    else:
        X.reset()
    covariance = streaming_low_rank_autocorrelation(
        X,
        rank,
        steps=steps,
        shift=mean,
        tree=tree,
    )
    return mean, covariance


# aliases
streaming_low_rank_covariance = streaming_low_rank_cov
streaming_mean_and_low_rank_covariance = streaming_mean_and_low_rank_cov
s_low_rank_autocorrelation = streaming_low_rank_autocorrelation
s_low_rank_cov = streaming_low_rank_cov
s_low_rank_covariance = streaming_low_rank_covariance
s_mean_and_low_rank_cov = streaming_mean_and_low_rank_cov
s_mean_and_low_rank_covariance = streaming_mean_and_low_rank_covariance
