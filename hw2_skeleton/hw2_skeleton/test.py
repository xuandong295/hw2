import numpy as np

X = np.array([[1], [3], [4]])
Y = np.array([[4], [5], [6]])
print(np.hstack((X,Y)))

A = X

degree = 3
if degree > 1:
    for i in range(2, degree+1):
        A = np.hstack((A, X**i))

print(A)