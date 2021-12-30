#최적해, 근사해, 예측값을 구하는 것 => LinearRegression
#b 벡터를 a벡터로 projection
#numpy : array, vector, matrix, 내적(dot product), 외적(outer product), matrix multiplication을 쉽게 수행하기 위해 사용
import numpy as np #dimension을 표현하기에 좋음
import plotly
import plotly.graph_objs as go

#b 벡터를 a벡터로 projection을 visualization해주는 함수 구현
#입력 매개변수는 벡터가 2개 있어야 하므로 두 개를 지정해준다.
def proj_to_line(vec_a, vec_b):


    # vector a를 3개로 분리 >>> plot을 하기 위해 각각 분리해준다.
    # a = [1, 2, 2]
    # a1 = 1, a2 = 2, a3 = 2

    a1, a2, a3 = vec_a
    b1, b2, b3 = vec_b


#함수로 입력되는 인자는 vector가 아닌 list로만 전달이 가능
#np를 이용해서 vector로 변경 후 넘겨도 됨
#함수 내에서는 리스트를 vector로 변경해서 사용(numpy의 array로 변환)
    vect_a = np.array(vec_a)
    vect_b = np.array(vec_b)


    #Projection Matrix: Outer Production / Inner Production(Dot Production)
    #v*v^T(외적) / v^T*v(내적) ==> Projection Matrix
    #v*v^T (두 개의 vector를 외적하면 matrix가 생성: 3x1 * 1x3 = 3x3)
    #v^T*v (두 개의 vector를 내적하면 실수(Real number) 생성: 1x3 * 3x1 = 1x1(R))

    P_a = np.outer(vect_a, vect_a) / vect_a.dot(vect_a)
    # print(P_a)

    #Projection Vector = Projection Matrix * b^T
    #Projection Vector = 3x1 (Projection Matrix 3x3 * 3x1 = 3x1)
    # p1, p2, p3로 분리해서 저장 (이유는 a vector와 b vector와 분리한 이유와 동일)
    p1, p2, p3 = P_a.dot(vect_b.T)

    data = []
    vector = go.Scatter3d(x=[0, a1], y=[0, a2], z=[0, a3],
                          marker=dict(size=[0, 5], color=['blue'],
                                      line=dict(width=5, color='DarkSlateGrey')),
                          name='a')


    data.append(vector)

    vector = go.Scatter3d(x=[0, b1], y=[0, b2], z=[0, b3],
                          marker=dict(size=[0, 5], color=['coral'],
                                      line=dict(width=5, color='DarkSlateGrey')),
                          name='b')

    data.append(vector)

    vector = go.Scatter3d(x=[0, p1], y=[0, p2], z=[0, p3],
                          marker=dict(size=[0, 5], color=['green'],
                                      line=dict(width=5, color='DarkSlateGrey')),
                          name='projection')

    data.append(vector)

    vector = go.Scatter3d(x=[b1, p1], y=[b2, p2], z=[b3, p3],
                          marker=dict(size=[0, 5], color=['violet'],
                                      line=dict(width=5, color='DarkSlateGrey')),
                          name='error')

    data.append(vector)

    fig = go.Figure(data=data)
    fig.show()



if __name__ == "__main__":
    proj_to_line([1, 2, 2], [1, 1, 1])







