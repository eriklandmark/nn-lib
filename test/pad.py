import numpy as np

a = np.array([[[[1, 7, 2], [11, 1, 23], [2, 2, 2]],
               [[1, 7, 2], [11, 1, 23], [2, 2, 2]]]])
print(np.pad(a, ((0,), (0,), (2,), (2, )), "constant"))