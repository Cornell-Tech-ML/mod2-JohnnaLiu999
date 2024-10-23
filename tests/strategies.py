from hypothesis import settings
from hypothesis.strategies import floats, integers


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def is_close(x: float, y: float) -> bool:
    """$f(x) = |x - y| < 1e-2$"""
    # ASSIGN 0.1
    return (x - y < 1e-2) and (y - x < 1e-2)
    # END ASSIGN 0.1


def assert_close(a: float, b: float) -> None:
    assert is_close(a, b), "Failure x=%f y=%f" % (a, b)
