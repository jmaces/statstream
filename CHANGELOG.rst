Changelog
=========

Versions follow `CalVer <https://calver.org>`_  (0Y.Minor.Micro format).

Changes for the upcoming release can be found in the `"changelog.d" directory <https://github.com/jmaces/statstream/tree/master/changelog.d>`_ in our repository.

..
   Do *NOT* add changelog entries here!

   This changelog is managed by towncrier and is compiled at release time.

   See our contribution guide for details.

.. towncrier release notes start

22.1.0 (2022-08-17)
-------------------

Changes
^^^^^^^

- ``statstream.approximate`` functionality is now exposed from the main module as described in the documentation.
  `#4 <https://github.com/jmaces/statstream/issues/4>`_


----


19.1.0 (2019-11-19)
-------------------

Changes
^^^^^^^

- Initial release.   `#3 <https://github.com/jmaces/statstream/issues/3>`_

  +  statstream.exact.streaming_min()
  +  statstream.exact.streaming_max()
  +  statstream.exact.streaming_mean()
  +  statstream.exact.streaming_var()
  +  statstream.exact.streaming_mean_and_var()
  +  statstream.exact.streaming_std()
  +  statstream.exact.streaming_mean_and_std()
  +  statstream.exact.streaming_cov()
  +  statstream.exact.streaming_mean_and_cov()

  + statstream.approximate.streaming_low_rank_autocorrelation()
  + statstream.approximate.streaming_low_rank_cov()
  + statstream.approximate.streaming_mean_and_low_rank_cov()
