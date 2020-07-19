import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt

def check_outliers(X):
    l = np.where(abs(X - np.mean(X)) >= 4 * np.std(X))
    return l[0].tolist()
    br = np.where(np.diff(*l)!=1)  # start loc
    delta = []
    dl = pd.to_timedelta(30, unit = 'm')
    try:
        if len(*br) == 0:
            if (time[l[0][-1]]-time[l[0][0]]) <= dl:
                delta = list(*l)
        else:
            br_idx = list(*map(lambda x: x+1, br))
            br_idx.insert(0,0)
            br_idx.append(len(*l))
            for i in range(len(br_idx)-1):
                    j = i+1
                    if (time[br_idx[j]]-time[br_idx[i]]) <= dl:
                        delta += list(l[0][i:j])
    except IndexError:
        delta = []
    return delta


def remove_outliers(y,X):
    ind1 = check_outliers(X)
    ind2 = check_outliers(y)
    ind = np.union1d(ind1, ind2).astype(int).tolist()
    X = np.delete(X,ind)
    y = np.delete(y,ind)
    return y,X

def check_constant(sequence):
    
    if len(sequence) == 0 or (np.count_nonzero(np.diff(sequence)==0)/len(sequence)) >= 0.99:  # unchange at more than 99% of the time
        sequence = np.append(sequence,'constant')
    return sequence


def tlcc_pearson(y, X, lag_limit=20):  # time-step is 90s due to rules of signal collection, upper bound of moving lag term is 30mins/90s=20
    clst=[]
    y,X = remove_outliers(y,X)
    y = check_constant(y)
    X = check_constant(X)
    for i in range(lag_limit): # 1 time-steps per movement
        if ((y[-1]=='constant')|(X[-1]=='constant')):    # return 99 if sequence is
            j=0
            clst.append(2)
            break
        x_lag = X[0:len(X)-1*i]
        y_lag = y[i:,]
        corr = round(stats.pearsonr(y_lag,x_lag)[0],3)
        clst.append(corr)
    #print(clst)
    j = np.argmax(np.absolute(clst))  # first occurance
    plot_image(correlation = clst,j=j)
    return j, clst[j]

def tlcc_spearman(y, X, lag_limit=20):  # time-step is 90s due to rules of signal collection, upper bound of moving lag term is 30mins/90s=20
    clst=[]
    y,X = remove_outliers(y,X)
    y = check_constant(y)
    X = check_constant(X)
    for i in range(lag_limit): # 1 time-steps per movement
        if ((y[-1]=='constant')|(X[-1]=='constant')):    # return 99 if sequence is
            j=0
            clst.append(2)
            break
        x_lag = X[0:len(X)-1*i]
        y_lag = y[i:,]
        corr = round(stats.spearmanr(y_lag,x_lag)[0],3)
        #corr = round(stats.pearsonr(np.nan_to_num((y_lag).astype(float)),np.nan_to_num((x_lag).astype(float)))[0],3)
        clst.append(corr)
    #print(clst)
    j = np.argmax(np.absolute(clst))  # first occurance
    #plot_image(correlation = clst, j = j)
    return j, clst[j],clst

def plot_image(correlation,j):
    plt.plot(range(len(correlation)),correlation,color = 'blue')
    plt.title(f'Peak is r ={correlation[j]} at lag = {j} ', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Correlation', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Lag')
    plt.show()

lag_limit = 20

#print(tlcc_pearson(y,X,lag_limit))
#print(tlcc_spearman(y,X,lag_limit))

variables = {'a':'Airflow','dp':'Damper Position','dat':'Discharge Air Temperature','zt':'Zone Temperature','as':'Airflow Setpoint','hwvc':'Hot Water Valve Command'}

#num = str(input("Select room:"))

var_1 = variables[input('variable one:')]
var_2 = variables[input('variable two:')]
df = pd.DataFrame(columns = ['Spearson','lag time','Correlation'])
for i in range(1,11):
    try:
        room = pd.read_csv('c:/Users/yanzheng/Downloads/10 Room Samples/Room-' +str(i)+'.csv')
    except FileNotFoundError:
        print('invalid room number entered')
    a = np.array(room[var_1].to_list())
    b = np.array(room[var_2].to_list())
    lag,corr,clst = tlcc_spearman(a,b,lag_limit)
    
    fig = plt.figure()
    plt.plot(range(len(clst)),clst,color = 'blue')
    plt.title('Room' + str(i) + " "+var_1 + " "+var_2+f':Peak is r ={corr} at lag = {lag} ', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Correlation', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Lag')
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.savefig('Room' + str(i) + " "+var_1 + " "+var_2+'.png',dpi = 180)
    #plt.show()

    if abs(corr) > 0.6:
        fd = 'Strong'
    elif abs(corr) < 0.6 and abs(corr) > 0.2:
        fd = 'Moderate'
    elif abs(corr) < 0.2:
        fd = 'Weak'
    else:
        fd = 'Constant Data'
    df.loc[str(i)] = [corr,lag,fd]
    df.index.name = 'Room Number'
print(df)
df.to_excel('Correlation' +var_1 + " "+var_2+'.xls')
#print(tlcc_pearson(a,b,lag_limit))