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

        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            value = f(*args, **kwargs)
            np.random.set_state(old_state)
            return value

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            for x in f(*args, **kwargs):
                yield x
            np.random.set_state(old_state)

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return set_seed_decorator
