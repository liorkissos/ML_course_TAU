### in order to generate files, Laod_DB needs to be ru first

import numpy as np
import matplotlib.pyplot as plt


### User Defined parameters
Ripple_max = 2 # in percent
eta = 0.0001

#### Data Loading and preperation
######################################
X_train= np.load('X_train.npy')
X_test = np.load('X_test.npy')
T_ind_train = np.load('T_ind_train.npy')
T_ind_test = np.load('T_ind_test.npy')

# concatenating a column of ones
X_train = np.concatenate((X_train,np.ones((X_train.shape[0],1))),axis=1)
X_train = X_train.T # so that the training index runs over columns.  (white)

X_test = np.concatenate((X_test,np.ones((X_test.shape[0],1))),axis=1)
X_test = X_test.T # so that the training index runs over columns.  (white)


# Problem parameters
K = 10 # number of classes
M = X_train.shape[0] # dimension of feature vector
N_train = X_train.shape[1] # length of training sequence
N_test = X_test.shape[1]


# Labels to one hot representarion
T_train = np.zeros([K,N_train])
T_train[T_ind_train.astype(int),np.arange(N_train)] = 1


####  Initializations
################################

W = np.random.rand(M,K) - 0.5

Y = np.zeros([K,N_train]) # y_l: the

Succes_rate_train_vec = []
Success_rate_test_vec = np.zeros(10)

Ripple = 100*np.ones(5)

iter=0

Ed_vec = []

#### Logistic Regression loop
##################################

while max(Ripple) > Ripple_max:

    denominator =np.sum(np.exp(W.T.dot(X_train)),axis = 0) # W:MxK .X_train :MxN_train. W.T.dotX_train: KxN_train. needs to be summed across the row dimension

    #for (w_l, t_l,y_l) in zip(W.T,T_train,Y):
    for l in np.arange(K): # running over the classes
        w_l = W[:,l] # W is MxK. w_l is Mx1
        t_l = T_train[l,:] # T_train is KxN_train. t_l is 1xN

        #denominator = np.sum(np.exp(w_l.T.dot(X_train)))  # X_train is MxN_train dimension

        y_l = np.exp(w_l.T.dot(X_train))/denominator # the logistic regression model for : Pr(c=c_k | x_n). y_l is 1xM * MxN_train = 1xN_train

        grad_Ed_l = (y_l-t_l).dot(X_train.T)           # gradient of Ed wrt w_l = sum_n((y_l-t_l)*x_n).  grad is 1xN_train * N_trainxM = 1xM
        w_l = w_l - eta*grad_Ed_l ## Gradient Descent

        Y[l, :] = y_l
        W[:,l] = w_l

    # result accumulation #1
    Ed  = -np.sum(T_train*np.log(Y))
    Ed_vec = np.append(Ed_vec, np.array(Ed))


    # Classification training
    Classification_train = Y.argmax(axis = 0)
    Failure_rate_train = np.count_nonzero(T_ind_train - Classification_train)*100/N_train
    Success_rate_train = 100 - Failure_rate_train
    print("Success rate train="+ str(Success_rate_train))
    Succes_rate_train_vec.append(Success_rate_train)

    # Classification test
    Y_test = np.exp(W.T.dot(X_test))/np.sum(np.exp(W.T.dot(X_test)),axis = 0)
    Classification_test = np.argmax(Y_test, axis = 0)
    Failure_rate_test = np.count_nonzero(T_ind_test - Classification_test) * 100 / N_test
    Success_rate_test = 100 - Failure_rate_test
    print("Success rate test ="+str(Success_rate_test))

    print("\n")

   # result accumulation #2
    Success_rate_test_vec = np.append(Success_rate_test_vec, np.array(Success_rate_test))

    # Termination condition
    Success_rate_test_ma = np.sum(Success_rate_test_vec[-10:])/10
    Ripple = np.abs(Success_rate_test_vec[-5:]-Success_rate_test_ma)*100/Success_rate_test_ma

    iter+=1


#### Display
######################

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Iteration #')
ax1.set_ylabel('Success rate over test set', color=color)
ax1.plot(np.arange(iter), Success_rate_test_vec[10:], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Cross Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(iter), Ed_vec, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()








