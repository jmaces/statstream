API Reference
=============

.. automodule:: statstream

What follows is the *API explanation*. This mostly just lists functions and
their options and is intended for *quickly looking up* things.

If you like a more *hands-on introduction*, have a look at our `examples`.


statstream.exact
----------------

.. automodule:: statstream.exact

Below is a list of all *exact* functions in the module.
They have aliases making them directly available from :py:mod:`statstream`.

.. autosummary::
    :toctree: api

    streaming_cov
    streaming_max
    streaming_mean
    streaming_mean_and_cov
    streaming_mean_and_std
    streaming_mean_and_var
    streaming_min
    streaming_std
    streaming_var


statstream.approximate
----------------------

.. automodule:: statstream.approximate

Below is a list of all *approximate* functions in the module.
They have aliases making them directly available from :py:mod:`statstream`.

.. autosummary::
    :toctree: api

    streaming_low_rank_autocorrelation
    streaming_low_rank_cov
    streaming_mean_and_low_rank_cov
