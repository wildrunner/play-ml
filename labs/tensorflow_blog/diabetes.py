from sklearn import datasets
from pylab import *

diabetes = datasets.load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)
print(diabetes.target[:10])
print(diabetes.data[:10])

plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
show()