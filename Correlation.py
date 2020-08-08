import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import random

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
    #plot_image(correlation = clst,j=j)
    return j, clst[j],clst

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
    fig = plt.figure()
    plt.plot(range(len(clst)),clst,color = 'blue')
    plt.title('Room' + str(i) + " "+var_1 + " "+var_2+f':Peak is r ={corr} at lag = {lag} ', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Correlation', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Lag')
    fig.set_size_inches(18.5, 10.5, forward=True)
    #fig.savefig('Room' + str(i) + " "+var_1 + " "+var_2+'.png',dpi = 180)
    #plt.show()
    #plt.show()

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn.dropna() , [25,75])
    IQR = Q3 - Q1
    
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    #print(lower_range,upper_range)
    return lower_range,upper_range

def read_in(var_1='a',var_2='dat',path = 0,limit = 1730,time_interval = 180):
    folderpath = r"C:\Users\yanzheng\Downloads\100 Rooms Data"
    filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)][0:20]
    path = filepaths[path]
    room = pd.read_csv(path)
    df_remarks = room[[var_1,var_2]]
    i = path[-10:-4]
    start = 0 
    end = 0
    method = 'pearson'
    if var_1 == 'Discharge Air Temperature' and var_2 == 'Hot Water Valve Command ':
        method
    while start < limit and end < limit:
        end = start + int(time_interval/5) 
        
        df_remarks= remark(var_1 = var_1, var_2 = var_2, start = start, end = end, df_remarks=df_remarks,method = method)
        start = end
    df_remarks.to_excel('room'+i+','+var_1 +','+ var_2+'.xls')
    return df_remarks
    #df = pd.DataFrame(columns = ['Correlation','Detection'])
def remark(var_1 = 'a',var_2 = 'dat',start = 0, end = 0,df_remarks = '',method = 'pearson'):
    dect = 0
    if 'result' not in df_remarks.columns:
        df_remarks['result'] = dect
    if 'corr' not in df_remarks.columns:
        df_remarks['corr'] = np.nan
    a = np.array(df_remarks[var_1][start:end+1].to_list())
    b = np.array(df_remarks[var_2][start:end+1].to_list())
    if method == 'pearson':
        lag,corr,clst = tlcc_pearson(a,b,lag_limit=20)
        if (var_1 == 'AirFlow' and var_2 == 'Airflow Setpoint') :
            for i in df_remarks.index.tolist():
                if abs(df_remarks['AirFlow'][i] - df_remarks['Airflow Setpoint'][i])/df_remarks['Airflow Setpoint'][i] > 0.1:
                    df_remarks['result'][i] = 1
                    df_remarks['corr'][i] = 0
                else:
                    df_remarks['result'][i] = 0
                    df_remarks['corr'][i] = 1
        elif (var_1 == 'Damper Position' and var_2 == 'AirFlow') or (var_1 =='Discharge Air Temperature' and var_2 == 'Zone Temperature'):
            if corr < 0.7:
                dect = 1
        elif var_1 == 'Discharge Air Temperature' and var_2 == 'Hot Water Valve Command ':
            if corr > -0.7:
                dect = 1
    elif method == 'spearman':
        lag,corr,clst = tlcc_spearman(a,b,lag_limit=20)
        if (var_1 == 'AirFlow' and var_2 == 'Airflow Setpoint') or (var_1 == 'Damper Position' and var_2 == 'AirFlow') or (var_1 =='Discharge Air Temperature' and var_2 == 'Zone Temperature'):
            if corr < 0.5:
                dect = 1
        elif var_1 == 'Discharge Air Temperature' and var_2 == 'Hot Water Valve Command ':
            if corr > -0.4:
                dect = 1

    df_remarks['corr'][start:end+1] = corr
    df_remarks['result'][start:end+1] = dect
    return df_remarks

    #if method == 'spearman':
    #    df.Correlation[df.Correlation== 2] = np.nan
   
   # result = df['Detection']
   # result.to_excel('Result'+var_1+','+var_2+'.xls')
    #faults.to_excel('Correlation' +var_1 + " "+var_2+ str(start)+':'+str(end)+'.xls')



