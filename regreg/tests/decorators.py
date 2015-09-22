import nose.tools as nt
import numpy as np

@nt.make_decorator
def set_seed_for_test(test_func):

    def test_func_with_seed_fixed(*args, **kwargs):
        old_state = np.random.get_state()
        np.random.seed(seed)
        test_func(*args, **kwargs)
        np.random.set_state(old_state)
    
    return test_func_with_seed_fixed
