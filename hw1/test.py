import numpy as np

# 定义两个张量
A = np.random.rand(2, 3, 2)  # Shape: (2, 3, 2)
B = np.random.rand(2, 3)     # Shape: (2, 3)


# 矩阵乘法
C = np.matmul(A, B)          
print(C.shape)               # Output: (2, 3, 3)


A = np.random.rand(2, 1, 3, 4)  # Shape: (2, 3, 4)
B = np.random.rand(3, 4, 5)  # Shape: (2, 4, 5)

C = np.matmul(A, B)          
print(C.shape)               # Output: (2, 3, 3, 5)
print(C)

A_batch = A[0, 0]  # Shape: (3, 4)
B_batch = B[0]        # Shape: (3, 4, 5)

C_batch = np.matmul(A_batch, B_batch)  # Shape: (3, 5)

C_check = np.matmul(A[1][0], B[1])
assert np.allclose(C[1, 1], C_check)
# 所以矩阵乘法广播的规则是什么呢？
# 1. 两个矩阵的最后两个维度要相等
# 2. 除了最后两个维度，其他维度的大小要相等或者其中一个为1
# 其他维度会被广播到相等的大小，然后逐个元素相乘，也就是逐个位置的矩阵相乘，类似标量的情形
