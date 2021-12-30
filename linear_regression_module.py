import torch
import torch.nn.functional as F #Loss function도 직접 구현, 사용 x
import torch.optim as optim
import torch.nn as nn
from Own_ML_Class import LogisticRegression

# 데이터 준비
x_data = [[1, 2], [2, 3], [3,1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# list to Tensor for training (using Torch)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

"""
모델 아키텍처 정의
nn.Sequential: layer를 차례로 쌓을 수 있도록 하는 것
"""
model = nn.Sequential(
    # nn.Linear : W, b값이 랜덤하게 셋팅된다.(자동 생성)
    nn.Linear(2, 1), #Wx + b (input dim, output dim) | MNIST : input: 28x28, output: 10(0~9까지)
    nn.Sigmoid()
) # tensorflow에도 존재



# 최적화 함수 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs + 1):
    """
    1. 가설함수 정의 및 계산 (forward)
    2. cost function 정의 및 계산
    3. w, b 값 업데이트 (using optmizer(미분)) : backward    
    """

    # 1. hypothesis function : sigmoid | H(x)
    # 예측값
    hypothesis = model(x_train) # x_train 데이터가 nn.Linear(2, 1), Sigmoid함수를 거친다.

    # binary cross entropy          |예측값, 실제값
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost 값을 최적화(minimize)하고 w, b 값 업데이트
    optimizer.zero_grad()
    cost.backward() # Loss(cost) function 미분 (기울기 계산)
    optimizer.step() # lr만큼 내려가면서 w , b 값 업데이트

    if epoch % 100 == 0:
        # 예측 값이 0.5를 넘으면 true
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        # print(prediction) # 예측값
        # print(correct_prediction) # 예측값과 Y_train 값이 같을 때 True를 반환한다.

        # accuracy : correct_prediction.sum() / 전체갯수 (6)
        # accuracy = correct_prediction.sum().item() >>> tensor(4), tensor(5), tensor(6)..
        # ==> 4 또는 5를 가지고 오기 위해서 .item() 사용 |tensor에 있는 값을 가지고오고 싶을 때 사용
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(accuracy)


        print(f'epoch : {epoch}/{nb_epochs}  cost: {cost.item()} Accuracy {accuracy * 100}%')




