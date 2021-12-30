import torch
import torch.nn as nn #nn.Linear 라이브러리를 사용하기 위해 import

# F.mse(mean squared error) <--linear regression, 다양한 LOSS Function 존재
# Classification problem에서 사용하는 loss function: Cross-Entropy (집합체 : torch.nn.functional에 존재한다.)
import torch.nn.functional as F
import torch.optim as optim #SGD, Adam, etc. 최적화 라이브러리

# 임의로 데이터 생성
# 입력이 1, 출력이 1
# Multi-variable linear regression(입력: 3, 출력 1)
# input(x_train) 4x3 2D Tensor
x_train = torch.FloatTensor([
    [11, 12, 13],
    [1, 2, 3],
[5, 6, 7],
[15, 26, 37]
])

#y_train (GT)
y_train = torch.FloatTensor([
    [58], #첫 번째 행의 출력값(1개)
[10],  #두 번째 행의 출력값(1개)
[24],  #세 번째 행의 출력값(1개)
[84],  #네 번째 행의 출력값(1개)
])
#모델 선언 및 초기화
# y = WX
# w = randn(1)....... <-- get_weights(1)
# model.parameters (weight: 3개,..., bias : 1)
# weight, bias: randomize 한 값으로 자동으로 셋팅
model = nn.Linear(3, 1) #get_weights()함수 참고..

#model.parameters() 최적화, w,b로 미분을 해야 하므로 requires_grad=True) 셋팅된 것을 확인할 수 있다.
#2D Tensor를 가지고 오는 이유는 row(batch size)를 통째로 가져와서 연산하기 위해서이다.
print(list(model.parameters()))

# optimizer = optim.SGD(model.parameters(), lr=1e-5) # 1e-5 == > 10^-5
optimizer = optim.SGD(model.parameters(), lr=0.00001) #lr ==> learning rate(조절하기 위해서는 lr을 낮춰주는 것이 좋다. ==> 보폭)

# dimension : cols



# epochs를 2000번 돌면서 random하게 들어가는 weight값과 x_train(입력 데이터)를 곱해준다.
nb_epochs = 2000
for epoch in range(nb_epochs+1):


    # H(x)를 구하는 부분 (forward)
    # weight * x_train(입력 데이터) 를 계산해서 y값을 도출하는 부분
    prediction = model(x_train)


    # cost 계산
    # mse = torch.mean((x_train - prediction).pow(2).sum())
    cost = F.mse_loss(prediction, y_train)

    # 최적의 w, b 값을 도출하기 위한 부분
    # cost를 최소화 하면서 (prediction-y_train) 값이 0에 수렴하는 것을 의미한다.

    # grad: 초기화하는 이유 : 누적이 발생하기 때문이다
    optimizer.zero_grad()

    # loss, cost function 함수를 미분하여 gradient 계산
    cost.backward() #미분한다 == gradient 계산(w, b 값 도출)

    # w, b
    # w.data -= lr * w.grad.data
    optimizer.step()

    if epoch % 100 == 0:        #4d ==> format(자릿수)
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

print(list(model.parameters()))

new_var = torch.FloatTensor([[11, 12, 13]])

# 152에 근접한 값이 출력이 되면 학습이 잘 된 것으로 판단.
pred_y = model(new_var) # model.forward(new_var)
print('After training: test(11, 12, 13) ==>', pred_y)







