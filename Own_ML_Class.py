import torch.nn as nn

# 학습할 모델을 정의하는 모듈
# Linear Regression
# Multivariable Linear Regression ==> (nn.Linear(n, 1))
# Logistic Regression (Binary Classification)


class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속 받음
    def __init__(self):
        # nn.Module 클래스의 속성들을 가지고 초기화
        super().__init__()
        # model = nn.Linear(1, 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, train_x):
        # model(train_x) or model.forward(train_x)
        return self.linear(train_x)  # Wx값 리턴


# bin_class = LogisticRegression(input_dim, output_dim)
# bin_class = LogisticRegression()

class LogisticRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()  # super : nn.Module에 존재하는 init 메소드를 모두(그대로) 사용하겠다는 의미
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim) # nn.Linear의 입력 인자 : in_dim, out_dim
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.linear(x))


    # model = LogisticRegression(2, 1)
    # model(x_train) is same as model.forward(x_train)





