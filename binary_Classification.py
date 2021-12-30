import torch
import torch.nn.functional #Loss function도 직접 구현, 사용 x
import torch.optim as optim

# 데이터 준비
x_data = [[1, 2], [2, 3], [3,1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# list to Tensor for training (using Torch)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 가설함수 (1/1+exp^-(wx+b))
# 6x2 * 2x1 = 6x1 ==> +b
# w, b값이 0이 되어도 실행이 되는 이유
# Logistic Regression hypothesis 함수는 sigmoid이기 때문에, 0으로 셋팅 시 초반 출력 값: 0.5

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 최적화 함수 설정
optimizer = optim.SGD([w, b], lr=1)

nb_epochs = 2000

for epoch in range(nb_epochs + 1):
    """
    1. 가설함수 정의 및 계산 (forward)
    2. cost function 정의 및 계산
    3. w, b 값 업데이트 (using optmizer(미분)) : backward    
    """

    # 1. hypothesis function : sigmoid | H(x)
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w)+b)))


    # 2. cost function 정의(Log 함수 이용)
    # 바이너리 classification 특성 상 (0~1 사이의 값이 출력이 되어야 하고, -log(x) {0<=x<=1}의 형태가
    # 구하고자 하는 loss에 적합함.
    cost = -(y_train * torch.log(hypothesis) + ((1-y_train) * torch.log(1-hypothesis))).mean()

    # cost 값을 최적화(minimize)하고 w, b 값 업데이트
    optimizer.zero_grad()
    cost.backward() # Loss(cost) function 미분 (기울기 계산)
    optimizer.step() # lr만큼 내려가면서 w , b 값 업데이트

    if epoch % 100 == 0:
        print(f'epoch : {epoch}/{nb_epochs}  cost: {cost.item()}')

print(w, b)

# 학습이 잘 되었으면
# 3개는 0에 가까운 값이 출력, 3개는 1에 가까운 값 출력

hypothesis = torch.sigmoid(x_train.matmul(w)+b)

pred = []
for i in list(hypothesis):
    if i >= 0.5:pred.append(1)
    else: pred.append(0)

print(pred)

# for문과 if문 없이 작성하기
# hypothesis : 2D-Tensor | FloatTensor : y_train값과 예측값을 비교해서 1D-Tensor로 나타내준다.
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)



