import math
from math import sin, cos, pi
import pandas as pd
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from surface_env import *
from matplotlib import animation
import collections


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        """
        if maxpooling layer is detected, only do maxpooling oepration itself
        don't have to use parameters such as weight and bias 
        """
        if layer['type']=='maxpool':
            continue
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.5

        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.5 + .5
    return params_values


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

"""
def relu(Z):
    return np.maximum(0,Z)
"""

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

"""
def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z < 0] = 0
    return dZ
"""


"""
because relu has vanishing gradient problem
I choose to use elu activation function instead (both forward and backward function)
so there will not be vanishing gradient problems
"""
def elu_forward(Z,alpha=0.1):
    return np.where(np.greater(Z,0),Z,alpha*(np.exp(Z)-1))

"""
dA is the derivative of cost function
Z is the weight matrix we need to do backward propagation
"""
def elu_backward(dA,Z,alpha=0.1):
    return np.where(np.greater(Z,0),dA,alpha*dA*np.exp(Z))


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        activation_func = elu_forward
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        """
        if maxpooling layer is detected, 
        I choose to use a dictionary to store the index for each data
        and this dictionary can be treated as a parameter in another dictionary named 'meomory'
        """
        if layer['type']=='maxpool':
            dic = collections.defaultdict(list)
            stride=layer['stride']
            diameter=layer['diameter']
            m=len(A_curr)
            n=len(A_curr[0])
            A_tmp = [[0]*n for _ in range(math.ceil(m/stride))]
            for j in range(n):
                for i in range(0,m,stride):
                    lo=i-diameter if i-diameter>0 else 0
                    hi=i+diameter if i+diameter<m-1 else m-1
                    tmp=[]
                    indices=[]
                    for k in range(lo,hi+1):
                        tmp.append(A_curr[k][j])
                        indices.append(k)
                    val=max(tmp)
                    A_tmp[i//stride][j]=val
                    index=tmp.index(val)
                    dic[j].append(indices[index])
            A_curr=np.array(A_tmp)
            memory['d'+str(idx)]=dic
            continue
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
    return A_curr, memory


def predict_once(x,y, params_values, nn_architecture):
  A_curr, memory = full_forward_propagation([[x],[y]], params_values, nn_architecture)
  return A_curr[0][0]


def get_cost_value(Y_hat, Y):

    m = Y_hat.shape[1]
    """
    I listed two cost functions here, one is commented
    The commented one is the cost function of cross entropy
    The one I choose to use for this assignment is MSE cost function
    """
    # cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    cost = 1 / m * np.sum((np.subtract(Y, Y_hat)) ** 2)
    return np.squeeze(cost)


def convert_prob_into_class(Y):
    return np.array([[1.0 if i > .5 else 0.0 for i in Y[0]]])


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = elu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)
    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, df):
    grads_values = {}

    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    """
    the foumula below is derivative of the original cost function, which is log loss (cross-entropy) cost function
    dA_prev =  - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    """
    dA_prev=df(Y, Y_hat)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        """
        when maxpooling layer is detected
        we can derive the index of each data 
        from the dictionary which is stored in 'memory' 
        """
        if layer['type']=='maxpool':
            dic=memory['d'+str(layer_idx_prev)]
            m=len(dA_prev)
            n=len(dA_prev[0])
            stride=layer['stride']
            tmp=[[0]*n for _ in range(m*stride)]
            # for j in range(n):
            #     for i in range(m):
            #         if i not in dic[j]:
            #             tmp[i][j]=0
            #         else:
            #             tmp[i][j]=dA_prev[i//stride][j]
            for j in range(n):
                for i in range(len(dic[j])):
                    tmp[dic[j][i]][j]=dA_prev[i][j]

            dA_prev=np.array(tmp)
            continue
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    print(grads_values)
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        """
        because we need to redesign the neural network
        same modification should be made here
        """
        if layer['type']=='maxpool':
            continue
        layer_idx += 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate, df):
    params_values = init_layers(nn_architecture, None)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        params_values, cost_history, accuracy_history, _ = train_once(X, Y, nn_architecture, learning_rate, params_values, cost_history, accuracy_history, df)
    return params_values, cost_history, accuracy_history


def train_once(X, Y, nn_architecture, learning_rate, params_values, cost_history, accuracy_history, df):
    Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
    cost = get_cost_value(Y_hat, Y)
    cost_history.append(cost)
    accuracy = get_accuracy_value(Y_hat, Y)
    accuracy_history.append(accuracy)

    grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture, df)
    params_values = update(params_values, grads_values, nn_architecture, learning_rate)
    return params_values, cost_history, accuracy_history, Y_hat


"""
visualization part should be modified which is based on maxpooling operation
I decide not to make comments for this part
just change several parameters below
"""
class OneStep(object):
    def __init__(self, X, Y, params_values, cost_history, accuracy_history, nn_architecture, df):
        self.params_values = params_values
        self.cost_history = cost_history
        self.accuracy_history = accuracy_history
        self.nn_architecture = nn_architecture
        self.Xs = X
        self.Ys = Y
        self.df=df

    def __call__(self):
        title="function"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.params_values, self.cost_history, self.accuracy_history, _ = train_once(self.Xs, self.Ys, self.nn_architecture, 1.0, self.params_values, self.cost_history, self.accuracy_history, self.df)
        #ax = plot(lambda i,j: predict_once(i,j, self.params_values, self.nn_architecture))
        #ax.clear()
        plt.title(title)
        ax.plot_surface(xcalc(x,y, lambda i,j: i), xcalc(x,y, lambda i,j: j), xcalc(x,y, lambda i,j: predict_once(i,j, self.params_values, self.nn_architecture)))
        ax.scatter(self.Xs[0], self.Xs[1], self.Ys, color="red")
        plt.show()


class AnimateTrainer:

    def init(self, X, Y, nn_architecture, epochs, learning_rate, ax, df):
        self.params_values = init_layers(nn_architecture, None)
        self.cost_history = []
        self.accuracy_history = []
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.counter = epochs
        self.nn_architecture = nn_architecture
        self.ax = ax
        self.df=df

    def __call__(self):
        self.counter -= 10
        for i in range(10):
            self.params_values, self.cost_history, self.accuracy_history, Y_hat = train_once(self.X, self.Y, self.nn_architecture, self.learning_rate, self.params_values, self.cost_history, self.accuracy_history, self.df)
        return self.X, self.Y, Y_hat, self.counter, self.params_values, self.nn_architecture, self.ax


def frames():
    iterater = 1
    while iterater > 0:
        X, Y, Y_hat, iterator, params_values, nn_architecture, ax= animate_train()
        yield [X, Y, params_values, nn_architecture, ax]


def animate(args):
    #fig.clear()
    ax.clear()
    ax.scatter(args[0][0], args[0][1], args[1][0], color="red")
    return ax.plot_surface(xcalc(x,y, lambda i,j: i), xcalc(x,y, lambda i,j: j), xcalc(x,y, lambda i,j: predict_once(i,j, args[2], args[3])))


def animate_n_train(X, Y, nn_architecture, epochs, learning_rate, df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0], X[1], Y[0], color="red")
    animate_train = AnimateTrainer()

    def frames():
        iterater = 1
        while iterater > 0:
            X, Y, Y_hat, iterator, params_values, nn_architecture, ax= animate_train()
            yield [X, Y, params_values, nn_architecture, ax]
        return None

    def animate(args):
        ax.clear()
        ax.scatter(args[0][0], args[0][1], args[1][0], color="red")
        return ax.plot_surface(xcalc(x,y, lambda i,j: i), xcalc(x,y, lambda i,j: j), xcalc(x,y, lambda i,j: predict_once(i,j, args[2], args[3])))

    animate_train.init(X, Y, nn_architecture, epochs, learning_rate, ax ,df)
    anim = animation.FuncAnimation(fig, animate, frames=frames)
    plt.show()
