import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from sklearn import preprocessing

class FFT():
    def __init__(self, num,var_1,var_2,a,b):
        self.num = num
        self.var_1 = var_1
        self.var_2 = var_2
        self.a = a
        self.b = b
      
    def check_outliers(self,X):
        l = np.where(abs(X - np.mean(X)) >= 4 * np.std(X))
        return l[0].tolist()

    def remove_outliers(self,y,X):
        X = np.array(X)
        y = np.array(y)
        ind1 = self.check_outliers(X)
        ind2 = self.check_outliers(y)
        ind = np.union1d(ind1, ind2).astype(int).tolist()
        X = np.delete(X,ind)
        y = np.delete(y,ind)
        return y,X

    def normalize_data(self,y,X):
        y,X = self.remove_outliers(y,X)
        d = np.array([y,X])
        #min_max_scaler = preprocessing.MinMaxScaler()
        #d = min_max_scaler.fit_transform(d)
        d = preprocessing.normalize(d)
        y = d[0]
        X = d[1]
        return y,X

    def fft_transform(self, a):
        a= np.subtract(a,np.mean(a))
        fft_a = fft(a)
        abs_a=np.abs(fft_a)  
        angle_a=np.angle(fft_a)  
        x = np.fft.fftfreq(len(a))
        return fft_a,abs_a,angle_a,x

    def plot_data(self,a,b):
      
       fft_a, abs_a,angle_a,x_1 = self.fft_transform(a)
       fft_b,abs_b,angle_b,x_2 = self.fft_transform(b)
       
       fig = plt.figure()
       
       #ax_1 = fig.add_subplot(2,1,1)
       #plt.subplot(211)
       plt.plot(x_1,abs_a,color = 'red')
       plt.plot(x_2,abs_b,color = 'blue')
       plt.title('Room'+self.num+':'+self.var_1+','+self.var_2)
       plt.xlabel('Frequency')
       plt.ylabel('Amplitude')

       #plt.subplot(212)
       #plt.plot(x_1,angle_a,color = 'red')
       #plt.plot(x_2,angle_b,color = 'blue')
       #plt.title('Room'+self.num+':'+self.var_1+','+self.var_2)
       #plt.xlabel('Frequency')
       #plt.ylabel('Phase')
       
       fig.set_size_inches(18.5, 10.5, forward=True)
       fig.savefig('Room'+str(self.num)+':'+self.var_1+','+self.var_2+'.png',dpi = 180)
       #plt.show()


if __name__ == '__main__':
    variables = {'a':'Airflow','dp':'Damper Position','dat':'Discharge Air Temperature','zt':'Zone Temperature','as':'Airflow Setpoint','hwvc':'Hot Water Valve Command'}
    #num = str(input("Select room:"))
    var_1 = variables[input('variable one:')]
    var_2 = variables[input('variable two:')]
    for i in range(1,11):
        try:
            room = pd.read_csv('c:/Users/yanzheng/Downloads/10 Room Samples/Room-' +str(i)+'.csv') 
            
        except FileNotFoundError:
            print('invalid room number entered')
        a=np.array(room[var_1].to_list())
        b=np.array(room[var_2].to_list())
        data = FFT(str(i),var_1,var_2,a,b)
        #data.plot_data(a,b)
        try:
            nor_a,nor_b = data.normalize_data(a,b)
   # print(nor_a)
   # print(nor_b)
        except ValueError:
            nor_a, nor_b = a,b
        
        fft_a, abs_a,angle_a,x_1 = data.fft_transform(a)
        fft_b,abs_b,angle_b,x_2 = data.fft_transform(b)
        fig = plt.figure()
        plt.plot(x_1,abs_a,color = 'red')
        plt.plot(x_2,abs_b,color = 'blue')
        plt.title('Room'+data.num+':'+data.var_1+','+data.var_2)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.savefig('Room' + str(i) + " "+var_1 + " "+var_2+'.png',dpi = 180)
       
 