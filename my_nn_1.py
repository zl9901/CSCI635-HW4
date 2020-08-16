import pandas as pd
import numpy as np
from np_nn import *


nn_architecture=[
    {"input_dim": 2, "output_dim": 4, "activation": "relu", "type": "dense"},
    {"input_dim": 4, "output_dim": 8, "activation": "relu", "type": "dense"},
    {"input_dim": 8, "diameter": 2, "stride": 2, "type": "maxpool"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid", "type": "dense"}
]


data = pd.read_csv("dataset1.csv")
Xs = np.array(data[["X0","X1"]]).T
Ys = np.array([data["Y"]])
learning_rate = 1.0


def dfMSE(y,yhat):
    return -2*np.subtract(y,yhat)
# def dfCrossEntropy(y,yhat):
#     return -(np.divide(y,yhat)-np.divide(1-y,1-yhat))

params_values, cost_history, accuracy_history = train(Xs,Ys,nn_architecture,4000,learning_rate,dfMSE)
onestep = OneStep(Xs, Ys, params_values, cost_history, accuracy_history, nn_architecture, dfMSE)
onestep()
print('final accuracy is '+str(accuracy_history[-1]))
