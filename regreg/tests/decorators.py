""" Decorators and helpers for the regreg test suite.
"""
import functools
import inspect

import numpy as np


def set_seed_for_test(seed=10):
    """
    Fix the seed for random test.

    Parameters
    ----------
    seed : int
        Random seed passed to np.random.seed

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, sets the
        random seed before running the test and then
        restores numpy's random state after running the test
        (even if the test raises).

    Notes
    -----
    The wrapper uses ``functools.wraps`` so that the test name, docstring
    and pytest marks of the decorated function are preserved.  The
    signature is also preserved (via ``__wrapped__``), so the decorator
    composes with ``pytest.mark.parametrize`` and fixtures.

    Generator functions (used as helpers yielding checks, not as tests)
    are also supported: the seed is set while the generator is consumed.
    """
    def set_seed_decorator(f):

        @functools.wraps(f)
        def setseed_func(*args, **kwargs):
            old_state = np.random.get_state()
            np.random.seed(seed)
            try:
                return f(*args, **kwargs)
            finally:
                np.random.set_state(old_state)

        @functools.wraps(f)
        def setseed_gen(*args, **kwargs):
            # generator helpers (e.g. ``Solver.test_*`` check generators):
            # the seed is in effect while the generator is being consumed
            old_state = np.random.get_state()
            np.random.seed(seed)
            try:
                yield from f(*args, **kwargs)
            finally:
                np.random.set_state(old_state)

        if inspect.isgeneratorfunction(f):
            return setseed_gen
        return setseed_func

    return set_seed_decorator
