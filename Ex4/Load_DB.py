
import numpy as np
from sklearn.datasets import fetch_mldata
#from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import urllib
from scipy.io import loadmat

import json

#### User Defined

Load_mode = "Internet"
#Load_mode = "File"

#### Script

if(Load_mode =="Internet"):

    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(
        mnist_alternative_url)

    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)

    mnist_raw = loadmat(mnist_path)

else:
    mnist_raw = loadmat('mnist-original.mat')


mnist = {"data": mnist_raw["data"].T, "target": mnist_raw["label"][0], "COL_NAMES": ["label", "data"],
         "DESCR": "mldata.orgdataset: mnist-original",}

# with open('file.txt','w') as file:
#     json.dump(mnist,file)

X = mnist['data'].astype('float64')
T_ind = mnist['target']
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
T_ind = T_ind[permutation]

# The next line flattens the vector into 1D array of size 784!
X = X.reshape((X.shape[0], -1))
X_train, X_test, T_ind_train, T_ind_test = train_test_split(X, T_ind, test_size=0.2)
scaler = StandardScaler()

#the next lines standardize the images
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save vectors into files
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('T_ind_train.npy', T_ind_train)
np.save('T_ind_test.npy', T_ind_test)

