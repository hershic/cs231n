import functools


def compose(*functions):
    """
    Creates a function composition (e.g. f(g(h(x))) given (f, g, h)) of the
    given functions, in which the innermost function is the rightmost list
    element.

    Inputs:
    - *functions: The functions to compose.

    Outputs:
    - reduction(): A function which applies the composition

      Inputs:
      - The input to the composition, e.g. x

      Outputs:
      - The result of the composition, e.g. f(g(h(x)))

    Example:
    - y = compose(f, g, h) returns a function which applies f(g(h(x))). We can
      call the reduction simply by calling y(x), given a compatible input x.
    """
    def _composed(f, g):
        return lambda x: f(g(x))

    return functools.reduce(_composed, functions, lambda x: x)


def chain(*functions):
    """
    Creates a function chain (i.e. h(g(f(x))) given (f, g, h)) of the given
    functions, in which the each element is applied, in turn, to the input,
    chaining the output from the previous function to the input of the next
    function.

    Inputs:
    - *functions: The functions to chain.

    Outputs:
    - reduction(): A function which applies the chain

      Inputs:
      - The input to the chain, e.g. x

      Outputs:
      - The result of the chain, e.g. h(g(f(x)))

    Example:
    - y = chain(f, g, h) returns a function which applies h(g(f(x))). We can
      call the reduction simply by calling y(x), given a compatible input x.
    """
    def _chained(f, g):
        return lambda x: g(f(x))

    return functools.reduce(_chained, functions, lambda x: x)
