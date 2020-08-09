#from Find_correlation import *
import torch
import numpy as np
from test import *
from sklearn import preprocessing
from count import *

def load_one_data(path = 0):
    folderpath = r"C:\Users\yanzheng\Downloads\100 Rooms Data"
    filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)][20:101]
    path = filepaths[path]
    room = pd.read_csv(path)
    df = room[[var_1,var_2]]
    df_remarks = normal_data(df.to_numpy())
    #print(df_remarks)
    #df_remarks =df_remarks.iloc[0:10]
    return df_remarks,df

#def load_all_data(path )

def load_input():
    df = test()
    df.dropna(inplace=True)
    
    labels = df['result'].to_numpy()#dependent variable
    labels = labels.T #transpose
    #print(type(labels))
    
    df.drop(columns = ['result','corr'], inplace = True)
    input_set = normal_data(df.to_numpy())
    #t = pd.read_csv('c:/Users/yanzheng/Desktop/samplesinput.csv',dtype=float)
    #labels = np.array(t['result'].tolist())
    #t = t[[var_1,var_2]]
    #input_set = normal_data(t.to_numpy())
    
    print(count(labels.tolist()))
    return labels,input_set

def normal_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data)
    return X_train_minmax
    


#
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def epoch(weights,bias,inputs,labels):
    
    for epoch in range(50): #times you train
        input_cal = inputs
        #print(np.shape(inputs))
        #print(np.shape(input_cal),np.shape(weights))
        XW = np.dot(input_cal, weights)+ bias
        z = sigmoid(XW)
        error = z - labels
    #print(error.sum())
        dcost = error
        dpred = sigmoid_derivative(z)
        z_del = dcost * dpred
        input_cal = input_cal.T
        weights = weights - lr*np.dot(input_cal, z_del)
    
    for num in z_del:
        bias = bias - lr*num
    return  bias, weights


#print(error.sum())

#slope = input x dcost x dpred


def nn(data,weights,bias,table):
    labels,inputs = load_input()
    #data['test result'] = 0
    list_data = []
    #dt = data.drop('test result',axis = 1)
    bias, weights = epoch(weights = weights,bias = bias,inputs = inputs,labels = labels)
    for i in data:
        #print(single_pt)
        result = sigmoid(np.dot(i, weights) + bias)
        #print(result)
        result = result[0]

        list_data.append(result)
        #print(list_data)
    table['test result'] = list_data
    #print(data)
    table.to_excel('data.xls')
    return table

if __name__ == '__main__':
    #Define Hyperparameters
    np.random.seed(42)
    weights = np.random.rand(2,1)
    bias = np.random.rand(1)
    lr = 0.05
    data,table= load_one_data() 
    
    #print(load_input()[0])
    #print(load_input()[1])
    nn(data=data,weights=weights,bias=bias,table=data)
#single_pt = np.array([1,0,0])
#result = sigmoid(np.dot(single_pt, weights) + bias)
#print(result)
