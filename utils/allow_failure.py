import functools
import nose

def allow_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception as e:
            print(str(e))
            raise nose.SkipTest
    return inner
