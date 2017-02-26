import functools


def compose(*functions):
    def composed(f, g):
        return lambda x: f(g(x))
    return functools.reduce(composed, functions[::-1], lambda x: x)
