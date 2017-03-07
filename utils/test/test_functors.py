import unittest

from utils.functors import chain, compose


class TestFunctorsBase(unittest.TestCase):
    def setUp(self):
        self.add = lambda i: (lambda x: x + i)
        self.sub = lambda i: (lambda x: x - i)
        self.mul = lambda i: (lambda x: x * i)
        self.div = lambda i: (lambda x: x / i)


class TestFunctorsChain(TestFunctorsBase):
    def test_two(self):
        self.assertEqual(chain(self.add(2), self.div(3))(10),
                         (10 + 2) / 3)

    def test_three(self):
        self.assertEqual(chain(self.add(2), self.div(3), self.mul(2))(10),
                         (10 + 2) / 3 * 2)


class TestFunctorsCompose(TestFunctorsBase):
    def test_two(self):
        self.assertEqual(compose(self.add(2), self.div(3))(10),
                         (10 / 3) + 2)

    def test_three(self):
        self.assertEqual(compose(self.add(2), self.div(3), self.mul(2))(10),
                         ((10 * 2) / 3) + 2)
