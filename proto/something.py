import numpy as np
from pprint import pprint

a = np.arange(16).reshape(2, 2, 4)
pprint(a)
b = a.transpose(0, 2, 1)
pprint(b)
pprint(b.shape)
c = np.reshape(b, (b.shape[0], -1))
pprint(c)
