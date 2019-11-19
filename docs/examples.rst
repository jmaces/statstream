Examples
========

Here we provide some more detailed examples showcasing some of the more advanced
aspects of using ``statstream``. For a simple use case and example of how to
get started we recommend to first have a look at our `overview` section.

.. contents:: Advanced topics addressed in our examples
    :depth: 1
    :local:
    :backlinks: none


Infinite Data Streams
---------------------

In some situations a stream of data might be infinite or just too long to be
completely consumed in an acceptable amount of time. Then even a single-pass
algorithm is not feasible. In these situations it is possible to specify a
maximal number of batches to extract from the stream of data to be used for
computing statistics. All following data will be ignored and does not affect
the statistics.

Let us first define an exemplary infinite data stream.

.. doctest:: INFINITESTREAMS

    >>> import numpy as np
    >>> def data_generator():
    ...     while True:
    ...         yield np.random.randn(2, 3)
    >>> data_stream = data_generator()

We can not simply call ``statstream.exact.streaming_mean(data_stream)`` to
compute the mean of the data stream, as the stream is not ending and this would
result in an infinite loop. Instead we can limit the number of batches
considered by passing the ``steps`` option.

.. doctest:: INFINITESTREAMS

    >>> from statstream.exact import streaming_mean
    >>> streaming_mean(data_stream, steps=10)  # doctest: +SKIP
    array([-0.16264697, -0.05351201,  0.02548519])  # random

Now only ten batches will be used to compute the mean. Depending on the data
stream the number of steps necessary to get good approximations to statistics
like, for example, the mean can differ quite considerably.

It is the users responsibility to never run streaming algorithms on an infinite
stream without specifying ``steps``. Since there is no way to determine whether
a data stream is infinite without consuming it, there is no way to prevent
infinite loops unless the user, who should know if the data stream is infinite
or not, provides the algorithm with this knowledge.


Resetting Data Streams
----------------------

Some streaming algorithms require to consume the stream of data more than once.
This is, for example, the case for computing low rank approximations to covariance
matrices. In a first pass the mean of the data set is computed exactly, which is then
used in a second pass to estimate the covariances.

For these situations it is necessary to be able to *restart* a stream of data. The ``statstream``
package provides two possibilities to achieve this.

a) A callable ``reset(X)`` is passed to the streaming algorithm that takes an ``Iterable`` as input
   and resets it to its initial state. The callable has to be provided by the user.
b) If no callble is passed to the streaming algorithm, it assumes that the ``Iterable`` itself provides
   a reset method. In other words, it will try to call ``X.reset()`` after each pass through the data.

Let us look at examples for both use cases.

Explicitly passing the ``reset()`` callable
```````````````````````````````````````````

We first set up an example data stream. For simplicity we store the full data set explicitly, but in a more
realistic example the data set is typically too large to be stored in memory and might have to be reloaded from files.

.. doctest:: RESETCALLABLE

    >>> import numpy as np
    >>> raw_data = np.random.randn(5, 2, 3)
    >>> def data_generator(dataset):
    ...     for i in range(dataset.shape[0]):
    ...         yield dataset[i, ...]
    >>> data_stream = data_generator(raw_data)

The ``raw_data`` contains five mini batches, each containing two 3d-vectors. The data stream iteratively provides these
five batches. Now we need to write the ``reset(X)`` function. Since in this simple example the full data set is stored
and thus never consumed, resetting the iterator is quite easy. We can just create a new ``Iterable`` from the raw data.

.. doctest:: RESETCALLABLE

    >>> def reset(X):
    ...     return data_generator(raw_data)

Now this can be used in any multi-pass algorithm.

.. doctest:: RESETCALLABLE

    >>> from statstream.approximate import streaming_low_rank_cov
    >>> cov_factor = streaming_low_rank_cov(data_stream, rank=2, reset=reset)
    >>> cov_factor  # doctest: +SKIP
    array([[-0.06645394, -0.42950694,  0.74912725],
           [-0.43654276, -0.71275165, -0.44737628]])  # random


Passing an ``iterable`` with internal ``reset()`` method
````````````````````````````````````````````````````````

We first set up an example data stream. For simplicity we store the full data set explicitly, but in a more
realistic example the data set is typically too large to be stored in memory and might have to be reloaded from files.

.. doctest:: RESETTABLEITERATOR

    >>> import numpy as np
    >>> raw_data = [np.random.randn(2, 3) for _ in range(5)]
    >>> class ResettableIterator(object):
    ...     def __init__(self, dataset):
    ...         self.list = dataset
    ...         self.iter = iter(self.list)
    ...
    ...     def __iter__(self):
    ...         return self.iter
    ...
    ...     def __next__(self):
    ...         return next(self.iter)
    ...
    ...     def reset(self):
    ...         self.iter = iter(self.list)
    >>> data_stream = ResettableIterator(raw_data)

The ``raw_data`` contains five mini batches, each containing two 3d-vectors. The data stream iteratively provides these
five batches. The ``ResettableIterator`` class implements the ``Iterable`` interface and additionally stores an internal
copy of the complete data set. Thus it can just create a new ``Iterable`` from the raw data, whenever its ``reset()`` method
is called.

.. doctest:: RESETTABLEITERATOR

    >>> next(data_stream)  # doctest: +SKIP
    array([[ 0.49718786, -0.15368937,  1.40891356],
           [ 1.70409972,  2.4122171 ,  1.9357452 ]])  # random
    >>> data_stream.reset()

Now this custom ``Iterable`` can be used in any multi-pass algorithm.

.. doctest:: RESETTABLEITERATOR

    >>> from statstream.approximate import streaming_low_rank_cov
    >>> cov_factor = streaming_low_rank_cov(data_stream, rank=2)
    >>> cov_factor  # doctest: +SKIP
    array([[-0.58551602, -0.56448936, -0.07790505],
           [ 0.7025856 , -0.78813722,  0.43026667]])  # random


If no ``reset(X)`` callable is passed to ``streaming_low_rank_cov()`` it will call the ``reset()`` method on ``data_stream`` instead.


Computing Covariance Matrices
-----------------------------

The simplest way to compute covariances matrices of a data stream is to use the ``streaming_cov()`` or ``streaming_mean_and_cov()``
functions from ``statstream.exact``.

.. doctest:: COVMATRICES

    >>> import numpy as np
    >>> from statstream.exact import streaming_cov
    >>> def data_generator(n):
    ...     for _ in range(n):
    ...         yield np.random.randn(2, 3)
    >>> data_stream = data_generator(5)
    >>> cov = streaming_cov(data_stream)
    >>> cov  # doctest: +SKIP
    array([[ 1.88289711, -0.12219587,  0.27254982],
           [-0.12219587,  0.87109854, -0.06253911],
           [ 0.27254982, -0.06253911,  0.82788062]])  # random


However, note that the size of the covariance matrix is the data dimension squared.

.. doctest:: COVMATRICES

    >>> cov.shape
    (3, 3)

This is typically vary large for many real data sets, for example image data.
Sometimes the full covariance matrix is not needed and an approximation is enough.
One possibility is to discard covariances and only compute the variances via ``streaming_var()``.
This basically computes the diagonal of the covariance matrix and ignores all off-diagonal entries.
Another possibility is to use a low rank factorization of the covariance matrix.
Covariance matrices are symmetric and positive semi-definite. Thus any covariance matrix :math:`A`
can be factorized as :math:`A=LL^\top`, for example using the Cholesky decomposition or singular value decomposition.
In fact if :math:`A=U\Sigma V^\top` is a singular value decomposition, we get :math:`U=V` due to symmetry
and thus :math:`L=U\Sigma^{1/2}` is a valid choice.
Discarding all but the largest :math:`k` singular values in :math:`\Sigma` and removing the corresponding columns
from :math:`U` we obtain a rank :math:`k` approximation to the covariance matrix :math:`A\approx U_k \Sigma_k U_k^\top = L_k L_k^\top`.

The ``statstream.approximate`` module provides streaming algorithm to compute these low rank factorizations of correlation and covariance matrices
from streaming data. Computing low rank covariance matrix factorizations is a two-pass algorithm and thus requires a resettable data stream.
See `Resetting Data Streams`_ for more details.

.. doctest:: COVMATRICES

    >>> from statstream.approximate import streaming_low_rank_cov
    >>> raw_data = np.random.randn(5, 2, 3)
    >>> def data_generator(dataset):
    ...     for i in range(dataset.shape[0]):
    ...         yield dataset[i, ...]
    >>> data_stream = data_generator(raw_data)
    >>> def reset(X):
    ...     return data_generator(raw_data)
    >>> cov_fac = streaming_low_rank_cov(data_stream, rank=2, reset=reset)
    >>> cov_fac  # doctest: +SKIP
    array([[-0.73538314, -0.02038839, -0.23446216],
           [ 1.33002195, -0.41501593, -0.9990144 ]])  # random


The size of a factor of the low rank factorization is only the rank times the data dimension instead of the data dimension squared.

.. doctest:: COVMATRICES

    >>> cov_fac.shape
    (2, 3)


Streaming Keras ``Sequences``
-----------------------------

A main motivation for ``statstream`` was the desire to analyze data sets encountered in various machine learning applications.
A frequently used framework for deep learning is Keras_, which provides its own ``Sequence`` class for streaming data.

.. _Keras: https://keras.io/

Let us see in a quick example how Keras ``Sequences`` work together with ``statstream``.
We begin by loading the MNIST image data set.

.. doctest:: KERASSEQUENCE
    :options: +SKIP

    >>> from keras.datasets import mnist
    >>> (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    >>> train_data.shape
    (60000, 28, 28)
    >>> train_labels.shape
    (60000,)

The training data set of MNIST contains 60000 examples of 28 x 28 pixel grayscale images.
The labels are integers encoding the ten possbile classes. We now make a ``Sequence`` from this data set.

.. doctest:: KERASSEQUENCE
    :options: +SKIP

    >>> import numpy as np
    >>> from keras.preprocessing.image import ImageDataGenerator
    >>> data_stream = ImageDataGenerator().flow(np.expand_dims(train_data, -1), batch_size=32)

The ``ImageDataGenerator`` class is useful for data augmentation and shuffling of a data set, however here we do not need this functionality.
The only important aspect for now is, that its ``flow()`` method returns a ``Sequence`` from our data set. This can now be used in ``statstream``
like any other ``Iterable``. The total number of samples in a ``Sequence`` and the size of the batches it provides can be accesed via
``data_stream.n`` and ``data_stream.batch_size`` respectively.
Note that we added an additional dimension to ``train_data`` before passing it to ``flow()``. This is necessary, because
the Keras ``ImageDataGenerator`` class expects image data to be of shape ``(num_samples, width, height, num_channels)``, where the number of channels is either
one for grayscale images, three for RGB images, or four for RGBA images.

.. doctest:: KERASSEQUENCE
    :options: +SKIP

    >>> from statstream.exact import streaming_mean
    >>> mean = streaming_mean(data_stream, steps=data_stream.n//data_stream.batch_size)
    >>> mean.shape
    (28, 28, 1)

.. warning::

    Keras ``Sequences`` are designed to be consumed many times, during the training of a neural network. Therefore, they automatically reset
    after being consumed. It is essential to provide the ``steps`` argument whenever using ``Sequences``, as otherwise the data stream
    consumed by the ``statstream`` function will never end. See `Infinite Data Streams`_ for more details.

Since Keras ``Sequences`` are resettable, they can easily be used with multi-pass algorithms.

.. doctest:: KERASSEQUENCE
    :options: +SKIP

    >>> from statstream.approximate import streaming_low_rank_cov
    >>> data_stream.reset()
    >>> cov_fac = streaming_low_rank_cov(data_stream, rank=16, steps=data_stream.n//data_stream.batch_size)
    >>> cov_fac.shape
    (16, 28, 28, 1)

.. tip::

    Often in machine learning applications we have data streams providing tuples of data batches, where each tuple contains
    both the actual training data, such as images, as well as their target labels. This is the case, if we provide labels to the
    ``flow()`` method of ``ImageDataGenerator``. For convenience, all ``statstream`` functions can handle tupled data. Only the first
    element of each tuple will be processed, all remaining elements are ignored. In other words, for ``Sequences`` yielding images and labels,
    ``statstream`` only processes the images, but not the labels.

.. doctest:: KERASSEQUENCE
    :options: +SKIP

    >>> data_stream = ImageDataGenerator().flow(np.expand_dims(train_data, -1), train_labels, batch_size=32)
    >>> imgs, labels = next(data_stream)
    >>> imgs.shape
    (32, 28, 28, 1)
    >>> labels.shape
    (32,)
    >>> data_stream.reset()
    >>> mean = streaming_mean(data_stream, steps=data_stream.n//data_stream.batch_size)
    >>> mean.shape
    (28, 28, 1)


Linear vs. Tree Combining Mode
------------------------------

This example explains a delicate detail of the algorithms for finding low rank approximations of correlation and covariance matrices.
Read the example on `Computing Covariance Matrices`_ first, since we assume that you are already familiar with computing low rank approximations
via singular value decompositions.

The streaming algorithms to compute these low rank approximations works by first computing singular value decompositions of the correlation or
covariance matrices of individual batches to obtain the low rank factorizations per batch. These are then combined in a clever way to obtain
approximate factorizations of the full data stream.

There are two ways of how to combine the batch-wise factorizations:

1. Batch-wise factorizations are combined in a *greedy* way, combining each new batch factorization with the current approximation of the
   previously already processed batches. We call this the *linear* combination mode, since at any time during the algorithm we only ever need to
   store two low rank factors. One for the currently processed batch and one accumulating the information of all already consumed batches. Hence,
   the memory requirement is linear in the size of the final approximate factorization.
2. Batch-wise factorizations are combined in a *pair-wise* way, always combining factorizations of two consecutive batches, then combining two of
   the already combined factorizations and so on, until finally all batches have been combined. We call this the *tree* combination mode, since
   the order of the processing and combining of batches can be visualized as a binary tree. This requires storing intermediate factorization of
   combined batches until they can be further combined. The memory required for this mode is the size of final approximate factor scaled by a factor
   logarithmic in the number of batches in the data stream.
   However, we have observed that often the approximations obtained this way are better than those obtained using the
   linear combination mode. Whether the increased accuracy justifies the increased memory usage (and whether this is even possible) depends on the
   individual application. We recommend using *tree* mode unless you have a good reason not to.

To demonstrate the usage of the two combination modes, we first create a resettable data stream as in `Resetting Data Streams`_.

.. doctest:: TREEMODE

    >>> import numpy as np
    >>> from statstream.approximate import streaming_low_rank_cov
    >>> raw_data = np.random.randn(5, 2, 3)
    >>> def data_generator(dataset):
    ...     for i in range(dataset.shape[0]):
    ...         yield dataset[i, ...]
    >>> def reset(X):
    ...     return data_generator(raw_data)

We now obtain a rank two approximation of the covariance matrix in linear mode.

.. doctest:: TREEMODE

    >>> data_stream = data_generator(raw_data)
    >>> cov_fac = streaming_low_rank_cov(data_stream, rank=2, reset=reset, tree=False)
    >>> cov_fac  # doctest: +SKIP
    array([[ 0.3865258 ,  0.15277284,  0.94745608],
           [ 0.55999836,  1.05657292, -0.39882531]])  # random

In this simple example with five batches, the combination order in linear mode is as follows:
First, batch one and two are combined.
Then the result is combined with batch three and so on.
Finally the combined result of the first four batches is combined with the fifth batch.
This is visualized below (left to right).

::

   batch1   batch2        batch3          batch4          batch5
        \     |             |               |               |
         \    |             |               |               |
        combined12 -- combined123 --  combined1234 -- combined12345


And now we do the same in tree mode.

.. doctest:: TREEMODE

    >>> data_stream = data_generator(raw_data)
    >>> cov_fac = streaming_low_rank_cov(data_stream, rank=2, reset=reset, tree=True)
    >>> cov_fac  # doctest: +SKIP
    array([[-0.73438567,  0.93960811,  0.10407417],
           [ 1.03639177,  0.88039226, -0.63524351]])  # random

In this simple example with five batches, the combination order in tree mode is as follows:
First, batch one and two are combined and the result is stored.
Then batches three and four are combined. The results of these two steps are combined next.
And finally the remaining fifth batch is combined with the accumulated result
of the first four batches.
This is visualized below (top to bottom).

::

    batch1  batch2  batch3  batch4  batch5
       \     /         \     /       /
        \   /           \   /       /
      combined12     combined34    /
             \         /          /
              \       /          /
            combined1234        /
                  \            /
                   \          /
                   combined12345
