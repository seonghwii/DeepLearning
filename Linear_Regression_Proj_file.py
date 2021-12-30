import numpy as np
from file_read import read_file
import torch

#Ax = b
#projection을 통해 근사값(최적해) 찾는 방법(w, bias)

house_info, cnt = read_file("sell_house.txt")

A = torch.FloatTensor(house_info)

x_train = A[:-5, 1:-1] #(23, 11)
y_train = A[:-5, -1:]  # 결과치 (23, 1)

x_test = A[-5:, 1:-1] #(5, 11)
y_test = A[-5:, -1:] #(5, 1)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)



if __name__ == '__main__':
    answer = []
    # x^ =(inv(A.T*A) * A.T) * b을 numpy를 사용하여 풀이.
    c= np.matmul(np.matmul(np.linalg.inv(np.matmul(x_test.T, x_test)), x_test.T), y_test) #matrix multi
    for i in x_test:
        sum = 0
        for j in range(len(i)):
            sum += c[j] * i[j]
        sum = float(sum)
        answer.append(sum)
    f = open("result.txt", "a")

    f.write('\nLinear_Algebra_Proj\n')
    for i in answer:
        f.write(f'{str(round(i, 2))}\n')

    f.close()


