import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

num_observation = 500
x1 = np.random.multivariate_normal([0, 0],[[1,.75], [.75, 1]], num_observation)
x2 = np.random.multivariate_normal([1, 4],[[1,.75], [.75, 1]], num_observation)

x = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observation), np.ones(num_observation))).tolist()

y_new = list(map(int, y))
colors = ['c', 'yellow']

x3 = np.ones((2*num_observation,1))
x_new = np.concatenate((x, x3), axis=1)


def f(x):
    return int(x > 0)


f_vector = np.vectorize(f)

weights = np.random.rand(1,3)

f_list = np.dot(x_new, weights.T)
f_result = f_vector(f_list)

darray = np.array(y_new)
darray = darray.reshape(-1, 1)

loss = np.sum(np.abs(darray - f_result))/f_result.shape[0]
rate = 0.001
while (loss > 0.005):
    delta = darray - f_result
    weights += rate*(np.dot(delta.T, x_new))
    f_result = f_vector(np.dot(x_new, weights.T))
    loss = np.sum(np.abs(darray - f_result))/f_result.shape[0]

print('Weights:', weights)


x_linear = np.linspace(-3, 3, 2*num_observation)
y_linear = (-1*weights[:,2]-weights[:, 0]*x_linear)/weights[:, 1]
plt.plot(x_linear, y_linear)

for i in range(2*num_observation):
    plt.scatter(x[i,0], x[i, 1], color=colors[y_new[i]])
plt.show()
