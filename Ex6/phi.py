
import numpy as np

def Phi_prime(x):
    #y = [(1/((1+np.exp**(-xx))**2) for xx in x]
   # y = [1/(1+np.exp(-xx)) for xx in x]

    y = 1/((1+np.exp(-x))**2)
    Phi_P = np.diag(y.squeeze()) # np.diag operates on array- like. i,e; (,n) and not (1,n). hence, we need to squeeze it
    return Phi_P