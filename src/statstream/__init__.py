from __future__ import absolute_import, division, print_function

from ._version_info import VersionInfo

# package meta data
__version__ = "19.1.0.dev"  # 0Y.Minor.Micro CalVer format
__version_info__ = VersionInfo._from_version_string(__version__)

__title__ = "statstream"
__description__ = "Statistics for Streaming Data"
__url__ = "https://github.com/jmaces/statstream"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Jan Maces"
__email__ = "janmaces[at]gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2019 {}".format(__author__)

# export main package functionality
__all__ = []
