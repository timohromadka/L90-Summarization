import numpy as np
import time

# =============================
# MISC FUNCTIONS
# =============================
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.5f} seconds to run.")
        return result
    return wrapper


# =============================
# FEATURIZER FUNCTIONS
# =============================