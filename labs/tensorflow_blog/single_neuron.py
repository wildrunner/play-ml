from sklearn import datasets
from pylab import *

diabetes = datasets.load_diabetes()


class SingleNeuron(object):
    def __init__(self):
        self._w = 0
        self._b = 0
        self._x = 0

    def set_params(self, w, b):
        """가중치와 바이어스 저장한다"""
        self._w = w
        self._b = b

    def forpass(self, x):
        """정방향 수식 w * x + b를 계산하고 결과를 리턴한다"""
        self._x = x
        y_hat = self._w * self._x + self._b
        return y_hat

    def backprop(self, err):
        """ 평균제곱오차를 사용한 공식 """
        m = len(self._x)
        self._w_grad = 0.1 * np.sum(err * self._x) / m
        self._b_grad = 0.1 * np.sum(err * 1) / m

    def update_grad(self):
        self.set_params(self._w + self._w_grad, self._b + self._b_grad)


n1 = SingleNeuron()
n1.set_params(5, 1)
print(n1.forpass(3))

for i in range(30000):
    y_hat = n1.forpass(diabetes.data[:, 2])
    error = diabetes.target - y_hat
    n1.backprop(error)
    n1.update_grad()
print('Final W', n1._w)
print('Final b', n1._b)
