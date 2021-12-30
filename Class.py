# 상속과 관련된 실습
import torch
import torch.nn as nn
import torch.nn.functional as F

class Computer:
    def __init__(self, cpu, ram):

        #parameter = arguments(할당)
        self.cpu = cpu
        self.ram = ram

    def browse(self):
        print("Web surfing...")

    def calc(self):
        print("Calculate CV, ML, DL...")


# computer 상속
# 특정 class를 상속받기 위해서는 클래스 명 뒤에 (상속받을 클래스 명)
class laptop(Computer):
    def __init__(self, cpu, ram, battery):
        # 자식 클래스에서 부모클래스의 내용을 그대로 사용하고 싶을 때
        # 컴퓨터 클래스의 __init__ 메소드를 그대로 사용하고 싶음.
        super().__init__(cpu, ram) #super를 사용하면 self.parameter는 적지 않아도 된다.
        # self.cpu = cpu
        # self.ram = ram
        self.battery = battery


    def move(self):
        print("노트북은 이동이 용이함.")

# Example#2 : Rectangle 넓이, 둘레, 등 구하기 (Rect > squared rect)
class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def perimeter(self):
        return 2 * (self.w + self.h)

    # area_tmp adding
    def area_tmp(self):
        print("inh-->inh 상속 --> 상속")
        return self.w * self.h



class Square:
    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w **2

    def perimeter(self):
        return 4 * self.w

class SquareInh(Rect):
    def __init__(self, w):
        super().__init__(w, w)

    def area_tmp(self):
        print("inh : 바로 위에 클래스 상속")
        return self.w * self.w


# super() vs. super(SquareInh. self)
# 탐색 범위가 달라짐 (다중 상속 또는 상속-> 상속일 경우)
# Rect, SquareInh (area_tmp 메소드를 이용해서 결과 확인)

class Cube(SquareInh):
    # 정육면체의 전체 면적 : w*h*6
    def surface_area(self):
        sur_area = super(SquareInh, self).area_tmp()
        return sur_area * 6

    def volumn(self):
        vol = super().area_tmp()
        return vol * self.w

# 단순 선형 회기 클래스 구현
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속 받음
    def __init__(self):
        # nn.Module 클래스의 속성들을 가지고 초기화
        super().__init__()
        # model = nn.Linear(1, 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, train_x):
        # model(train_x) or model.forward(train_x)
        return self.linear(train_x)  # Wx값 리턴


if __name__ == '__main__':
    laptop = laptop("Intel CPU 2G", "8GB", "100%")
    laptop.browse(), laptop.move()


    squar_inh = SquareInh(4)
    # 예상하는 넓이와 둘레의 결과 값 : 넓이(16), 둘레(16)
    print(squar_inh.area(), squar_inh.perimeter())

    cube = Cube(3)
    print(cube.surface_area())
    print(cube.volumn())

    X_train = torch.FloatTensor([[1], [2], [3]])
    Y_train = torch.FloatTensor([[1], [2], [3]])

    # previous version : model = nn.Linear(1,1)
    # 객체를 생성한다는 것 이외에는 동일함.
    model = LinearRegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    nb_epochs = 2000

    for epoch in range(nb_epochs):
        pred = model(X_train)

        cost = F.mse_loss(pred, Y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            tmp = list(model.parameters())
            print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

    print(f'W, b: {tmp[0], tmp[1]}')


            # 실제 학습시키는 부분은 딥러닝 수학 시간에 했던 것을 참고해서 동작 여부 확인 해보기.




# a = Computer("1.6", "86") # => cpu, ram에 각 값이 들어간다.



