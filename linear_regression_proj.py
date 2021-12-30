import numpy as np

#Ax = b
#projection을 통해 근사값(최적해) 찾는 방법(w, bias)
A = np.array([[1, 1], [2, 1], [3, 1]]) # 2D-Tensor
b = np.array([2, 4, 6]) # vector_1D Tensor

# projection을 이용해서 근사해 구하는 공식
# x^ =(inv(A.T*A) * A.T) * b
# x^ : 구하고자 하는 최적해
# * : matrix multiplication
# .T : transpose
# inv(inverse matrix) : np.linalg.inv (numpy에서 제공하는 inverse matrix 구하는 라이브러리)
# matrix A = 3x2
# transpose(3x2) ==> 2x3 * 3x2 ==> 2x2 ==> inv(2x2) ===> 2x2 * 2x3(2번째 A.T)
# ==> 2x3 * 3x1(y_train) ==> 2x1(weight, bias)

# 역행렬 구하기 위한 조건
# 1. square matrix(2x2)
# *2. det != 0이어야 한다.
# 3. 선형 독립이어야 한다. == 역행렬을 구하고자 하는 matrix의 rank는 full rank를 가져야 한다.

c = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b) #matrix multi
print(c)
print(f'weight={c[0]}, bias={c[1]}')









