import unittest
import numpy as np

from regularizers.l2 import RegularizerL2


class TestRegularizerL2Directed(unittest.TestCase):
    def setUp(self):
        self.regularizer = RegularizerL2()

    def testRegularizerBasic(self):
        self.weights = np.array([[1, 2, 3, 2],
                                 [2, 4, 2, 3],
                                 [3, 1, 2, 4]])
        self.regularizer.addLayer(self)
        self.assertAlmostEqual(self.regularizer.calculate(), 40.5)

    def testRegularizerPrefersL2(self):
        self.regularizer.addLayer(self)
        self.weights = np.array([[.25, .25, .25, .25],
                                 [.25, .25, .25, .25],
                                 [.25, .25, .25, .25]])
        distributedWeights = self.regularizer.calculate()
        distributedWeightsSum = np.sum(self.weights)

        self.weights = np.array([[0, 0, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 1]])
        localizedWeights = self.regularizer.calculate()
        localizedWeightsSum = np.sum(self.weights)

        self.assertAlmostEqual(distributedWeightsSum, localizedWeightsSum)
        self.assertLess(distributedWeights, localizedWeights)
