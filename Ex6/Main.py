

import numpy as np
import functools as func
import itertools as it
import scipy.special as sc

from phi import *

### Creating database

mu = 0.001
L = 3

X = list(it.product([0,1], repeat = L))
X = np.array(X)

T = []
for i,x  in enumerate(X):
    t = func.reduce(lambda a,b: a^b, x)
    T.append(t)

##### initializations

W1 = np.ones((L,L))
b1 = np.ones((L,1))

W3 = np.ones((L,1))
b3 = np.ones((1,1))


for x,t in zip(X,T):

    Theta = np.concatenate((W1.flatten(),b1.flatten(),W3.flatten(), b3.flatten()),axis = -1)

    x= x.reshape((L,1))

    m1 = x # 3x1
    m2 = W1.T.dot(m1)+b1 # 3x1
    m3 = sc.expit(m2) # 3x1
    m4 = W3.T.dot(m3)+ b3 # 1x1
    m5 = (m4 - t)**2 # 1x1

    delta5 = np.array([[1]]) # 1x1
    delta4 = delta5.T.dot(2*(m4 - t)) # 1x1
    delta3 = delta4.dot(W3.T) # 3x1
    delta2 = delta3.dot(Phi_prime(m2)) # 1x3 * 3x3 = 1x3

    dE_dW3 = delta4.dot(m3.T)
    dE_db3 = delta4.dot(np.eye(1))

    dE_dW1 = delta2.T.dot(m2.T)
    dE_db1 = delta2.dot(np.eye(L))

    dE_d_Theta = np.concatenate((dE_dW1.flatten(),dE_db1.flatten(),dE_dW3.flatten(),dE_db3.flatten()), axis = -1)

    Theta = Theta - mu*dE_d_Theta

    a = 1







