#from Find_correlation import *
import torch
import numpy as np
from test import *

def load_one_data(path = 0):
    folderpath = r"C:\Users\yanzheng\Downloads\100 Rooms Data"
    filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)][20:101]
    path = filepaths[path]
    room = pd.read_csv(path)
    df_remarks = room[[var_1,var_2]]
    df_remarks =df_remarks.iloc[0:10]
    return df_remarks

#def load_all_data(path )

def load_input():
    df = test()
    df.drop(columns=['corr'], inplace=True)
    
    labels = df['result'].to_numpy()#dependent variable
    labels = labels.T #transpose

    df.drop(columns = ['result'], inplace = True)
    input_set = df.to_numpy() 
    return labels,input_set



#
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def epoch(weights,bias):
    for epoch in range(25000):
        labels,inputs = load_input()
        XW = np.dot(inputs, weights)+ bias
        z = sigmoid(XW)
        error = z - labels
    #print(error.sum())
        dcost = error
        dpred = sigmoid_derivative(z)
        z_del = dcost * dpred
        inputs = inputs.T
        weights = weights - lr*np.dot(inputs, z_del)
    
    for num in z_del:
        bias = bias - lr*num
    return  bias, weights


#print(error.sum())

#slope = input x dcost x dpred


def nn(data,weights,bias):
    bias, weights = epoch(weights = weights,bias = bias)
    for i in data.to_numpy():
        single_pt = np.array(i)
        result = sigmoid(np.dot(single_pt, weights) + bias)
        data['test result'][i] = result
    print(data)
    return data

if __name__ == '__main__':
    #Define Hyperparameters
    np.random.seed(42)
    weights = np.random.rand(2,1)
    bias = np.random.rand(1)
    lr = 0.05
    data = load_one_data() 
    nn(data,weights,bias)
#single_pt = np.array([1,0,0])
#result = sigmoid(np.dot(single_pt, weights) + bias)
#print(result)

