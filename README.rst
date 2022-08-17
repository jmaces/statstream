=============================================
``statstream``: Statistics for Streaming Data
=============================================

.. add project badges here
.. image:: https://readthedocs.org/projects/statstream/badge/?version=latest
    :target: https://statstream.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/jmaces/statstream/actions/workflows/pr-check.yml/badge.svg?branch=master
    :target: https://github.com/jmaces/statstream/actions/workflows/pr-check.yml?branch=master
    :alt: CI Status

.. image:: https://codecov.io/gh/jmaces/statstream/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jmaces/statstream
  :alt: Code Coverage

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style: Black


.. teaser-start

``statstream`` is a lightweight Python package providing data analysis and statistics utilities for streaming data.

Its main goal is to provide **single-pass** variants of conventional `numpy <https://numpy.org/>`_
data analysis and statistics functionality for **streaming** data that is
either generated on the fly or to large to be handled at once. Data can be
streamed as in chunks called **mini-batches**, which makes ``statstream``
extremely useful in combination with machine learning and deep learning
packages like `keras <https://keras.io/>`_, `tensorflow <https://www.tensorflow.org/>`_, or `pytorch <https://pytorch.org/>`_.

.. teaser-end


.. example

``statstream`` functions consume iterators providing batches of data.
They compute statistics of these batches and combine them to obtain statistics
for the full data set.

.. code-block:: python

   import statstream
   mean = statstream.streaming_mean(some_iterable)

The `Overview <https://statstream.readthedocs.io/en/latest/overview.html>`_ and
`Examples <https://statstream.readthedocs.io/en/latest/examples.html>`_ sections
of our documentation provide more realistic and complete examples.

.. project-info-start

Project Information
===================

``statstream`` is released under the `MIT license <https://github.com/jmaces/statstream/blob/master/LICENSE>`_,
its documentation lives at `Read the Docs <https://statstream.readthedocs.io/en/latest/>`_,
the code on `GitHub <https://github.com/jmaces/statstream>`_,
and the latest release can be found on `PyPI <https://pypi.org/project/statstream/>`_.
It’s tested on Python 2.7 and 3.5+.

If you'd like to contribute to ``statstream`` you're most welcome.
We have written a `short guide <https://github.com/jmaces/statstream/blob/master/.github/CONTRIBUTING.rst>`_ to help you get you started!

.. project-info-end


.. literature-start

Further Reading
===============

Additional information on the algorithmic aspects of ``statstream`` can be found
in the following works:

- Tony F. Chan & Gene H. Golub & Randall J. LeVeque,
  “Updating formulae and a pairwise algorithm for computing sample variances”,
  1979
- Radim, Rehurek,
  “Scalability of Semantic Analysis in Natural Language Processing”,
  2011

.. literature-end


Acknowledgments
===============

During the setup of this project we were heavily influenced and inspired by
the works of `Hynek Schlawack <https://hynek.me/>`_ and in particular his
`attrs <https://www.attrs.org/en/stable/>`_ package and blog posts on
`testing and packaing <https://hynek.me/articles/testing-packaging/>`_
and `deploying to PyPI <https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/>`_.
Thank you for sharing your experiences and insights.
