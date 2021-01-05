"""Statistics utilities for streaming numpy data.

This package provides several utilities for computing statistics of
univariate or multivariate data from samples.

Unlike the corresponding `numpy` functions (`numpy.mean`, `numpy.var`,
`numpy.cov` etc.) this package is desgined to work with a
*stream of mini-batches* of samples instead of the full dataset at once.
This is particularly useful for very large data sets that can not be completely
stored in memory.

The package is organized in two modules.

- `statstream.exact` contains exact streaming version of the corresponding
  numpy functions. This includes simple statistics like mean, variance,
  minimum, and maximum.
- `statstream.approximate` contains approximate functions for more complex
  statistics that can not be computed from streaming data in a single pass.
  This includes for exmaple low rank factorisations of covariance matrices.

All functions in this package are named like the corresponding non-streaming
`numpy` functions (if they exists) except for the prefix ``streaming_``.
For brevity also aliases with the shorter prefix ``s_`` are provided.

"""
from __future__ import absolute_import, division, print_function

from .approximate import (  # noqa:F401
    s_low_rank_autocorrelation,
    s_low_rank_cov,
    s_low_rank_covariance,
    s_mean_and_low_rank_cov,
    s_mean_and_low_rank_covariance,
    streaming_low_rank_autocorrelation,
    streaming_low_rank_cov,
    streaming_low_rank_covariance,
    streaming_mean_and_low_rank_cov,
    streaming_mean_and_low_rank_covariance,
)
from .exact import (  # noqa:F401
    s_cov,
    s_covariance,
    s_max,
    s_mean,
    s_mean_and_cov,
    s_mean_and_covariance,
    s_mean_and_std,
    s_mean_and_var,
    s_mean_and_variance,
    s_min,
    s_std,
    s_var,
    s_variance,
    streaming_cov,
    streaming_covariance,
    streaming_max,
    streaming_mean,
    streaming_mean_and_cov,
    streaming_mean_and_covariance,
    streaming_mean_and_std,
    streaming_mean_and_var,
    streaming_mean_and_variance,
    streaming_min,
    streaming_std,
    streaming_var,
    streaming_variance,
)


# package meta data
__version__ = "19.2.0.dev0"  # 0Y.Minor.Micro CalVer format
__title__ = "statstream"
__description__ = "Statistics for Streaming Data"
__url__ = "https://github.com/jmaces/statstream"
__uri__ = __url__
# __doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Jan Maces"
__email__ = "janmaces[at]gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright 2019 Jan Maces"
