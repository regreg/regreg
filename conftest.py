"""pytest configuration for the regreg test suite.

Provides:

* ``--runslow`` command line option; tests marked ``@pytest.mark.slow``
  are skipped unless it is given.
* ``--seed`` option and an ``rng`` fixture returning a seeded
  ``np.random.Generator``. Tests draw random numbers from their own
  ``np.random.default_rng`` generators, never from numpy's global RNG.
  (Fixtures defined here do not reach the tests when they are run from
  the installed package, as in CI, so tests should not rely on them.)
* automatic skipping of tests marked ``rpy2`` (comparisons with R) when
  ``rpy2`` or a working R installation is unavailable.
* ``collect_ignore`` for sandbox directories.
"""
import functools
import os

import numpy as np
import pytest

# Directories / files that are not part of the test suite.
collect_ignore = [os.path.join("regreg", "tests", "sandbox"),
                  "sandbox",
                  "doc",
                  "tools",
                  "build"]
collect_ignore_glob = ["*/sandbox/*", "**/sandbox"]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked as slow")
    parser.addoption("--seed", action="store", type=int, default=0,
                     help="seed for the rng fixture")


@functools.lru_cache(maxsize=None)
def _have_rpy2():
    """Return (available, reason) for rpy2 + a working R."""
    try:
        import rpy2.robjects as robjects  # noqa: F401  (starts embedded R)
        robjects.r("1")
    except Exception as e:  # ImportError, RRuntimeError, missing R_HOME...
        return False, "rpy2/R not available (%s: %s)" % (type(e).__name__, e)
    return True, ""


def pytest_collection_modifyitems(config, items):
    runslow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow test: use --runslow to run")
    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if item.get_closest_marker("rpy2") is not None:
            ok, reason = _have_rpy2()
            if not ok:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def rng(request):
    """A seeded ``np.random.Generator``."""
    return np.random.default_rng(request.config.getoption("--seed"))
