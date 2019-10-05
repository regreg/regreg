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
        restores numpy's random state after running the test.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.


    """
    import nose
    def set_seed_decorator(f):

        def setseed_func(*args, **kwargs):
            """Setseed for normal test functions."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            value = f(*args, **kwargs)
            np.random.set_state(old_state)
            return value

        def setseed_gen(*args, **kwargs):
            """Setseed for test generators."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            for x in f(*args, **kwargs):
                yield x
            np.random.set_state(old_state)

        # Choose the right setseed to use when building the actual decorator.
        if nose.util.isgenerator(f):
            setseed = setseed_gen
        else:
            setseed = setseed_func

        return nose.tools.make_decorator(f)(setseed)

    return set_seed_decorator
