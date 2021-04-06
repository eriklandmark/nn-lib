def heapPermutation(a, size):
    if size == 1:
        print(a)
        return

    for i in range(size):
        heapPermutation(a, size-1)
        if size % 2 != 0:
            a[0], a[size-1] = a[size-1], a[0]
        else:
            a[i], a[size-1] = a[size-1], a[i]


# Driver code
a = [1, 2, 3]
n = len(a)
heapPermutation(a, n)