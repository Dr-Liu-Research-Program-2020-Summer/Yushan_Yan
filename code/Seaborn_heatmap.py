import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import seaborn as sn
import matplotlib.pyplot as plt

for i in range(1,11):
    room = pd.read_csv('c:/Users/yanzheng/Downloads/10 Room Samples/Room-' +str(i)+'.csv')

    Spearman = round(room.corr(method='spearman',min_periods=1),2)   # method默认Spearman
    print(Spearman)
    fig = plt.figure()
    ax = sn.heatmap(Spearman, annot=True)
    ax.set_title('Room'+str(i))
    fig.savefig('Room'+str(i)+'_Heatmap.png',dpi = 180)