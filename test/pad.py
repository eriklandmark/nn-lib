import numpy as np

a = np.array([[[[1, 7, 2], [11, 1, 23], [2, 2, 2]],
               [[1, 7, 2], [11, 1, 23], [2, 2, 3]]]
              ,[[[1, 7, 2], [11, 1, 23], [2, 2, 2]],
                [[1, 7, 2], [11, 1, 23], [2, 2, 3]]]])
b = np.array([[[[1, 5, 9], [2, 6, 10]], [[3, 7, 11], [4, 8, 12]]]])

c = np.array([[1,2,3], [3,4,5]])

#[[[13, 17, 21], [14, 18, 22]], [[15, 19, 23], [16, 20, 24]]]

print(a.shape)
print(np.pad(a, ((0,), (0,), (2,), (2, )), "constant"))
print(np.sum(a, axis=(0,2,3)))

print(b.shape)
print(np.sum(b, axis=(0, 2,3)))


print(np.mean(c, axis=0))
print(np.mean(c, axis=1))
