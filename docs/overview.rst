Overview
========

``statstream`` provides functions for obtaining statistical insights from a
stream of data. We think it is best to show the core concepts of the package by
a simple exemplary demonstration.

Let us begin by synthetically generating a stream of random data.

.. doctest::

    >>> import numpy as np
    >>> def data_generator(n):
    ...     for _ in range(n):
    ...         yield np.random.randn(2, 3)
    >>> data_stream = data_generator(8)

We have created a generator function creating a stream of random data of shape
``(2, 3)``. The first axis is considered as ``batch_size``, the remaining axes
as data dimensions. So here, or stream provides chunks of 3-dimensional
vectors, each chunk containing two such vectors. Calling ``data_generator(n)``
with ``n=8`` we obtain an iterator providing eight such chunks.

Let us have a look at one such chunk.

.. doctest::

    >>> next(data_stream)  # doctest: +SKIP
    array([[-0.96083597, -0.86513521, -0.70060355],
           [ 0.2771605 , -1.58573487,  1.16854072]])  # random


The full data set contains sixteen vectors (eight chunks of two vectors each).
In this simple example we could read out the full iterator into a single array
and compute statistics like component-wise mean or variance for the data set.


.. doctest::

    >>> arr = np.concatenate(list(data_stream), axis=0)
    >>> arr.shape
    (16, 3)
    >>> np.mean(arr, axis=0)  # doctest: +SKIP
    array([ 0.30680563, -0.30335616, -0.06189916])  # random
    >>> np.var(arr, axis=0) # doctest: +SKIP
    array([0.93472631, 1.4720951 , 1.13723147])  # random

However, if the data set iterator can not be transformed into a list, for
example because it is too large to be stored in memory or generated in
real-time, computing mean and variance in this way is impossible. Instead we
need to use an algorithm that can handle streaming data. This is what
``statstream`` is for.

In the above example we have exhausted our streaming data source, so
``data_stream`` is now empty. Lets quickly generate a new stream and compute
the mean from it.

.. doctest::

    >>> data_stream = data_generator(8)
    >>> from statstream.exact import streaming_mean, streaming_var
    >>> streaming_mean(data_stream) # doctest: +SKIP
    array([-0.41722521,  0.0331529 ,  0.14349293])  # random

The same can of course also be done for the variance.

.. doctest::

    >>> data_stream = data_generator(8)
    >>> streaming_var(data_stream)  # doctest: +SKIP
    array([0.64355842, 1.44646403, 0.74006582])  # random

Note that again we have to create a new ``data_stream``, since the other one
was consumed by ``streaming_mean`` and afterwards empty.
