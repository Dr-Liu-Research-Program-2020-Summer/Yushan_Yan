import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt

for i in range(1,100):
    room = pd.read_csv('c:/Users/yanzheng/Downloads/10 Room Samples/Room-' +str(i)+'.csv')

    Spearman = round(room.corr(method='spearman',min_periods=1),2)   # method默认Spearman
    print(Spearman)

    Spearman_abs = np.abs(Spearman)
    Spearman_abs.style.background_gradient(cmap='Blues',axis =1,low=0,high=1)

    print(Spearman_abs)
    fig,ax = plt.subplots(1,1,figsize=(16,12))
    hot_img = ax.matshow(np.abs(Spearman),vmin=0,vmax=1,cmap='Greens')
# vmin=0,vmax=1  设置值域从0-1
    fig.colorbar(hot_img)  # 生成颜色渐变条(右侧)
    ax.set_title('Room'+str(i)+' Heatmap-Spearman',fontsize=14,pad=12)
    ax.set_xticks(range(0,6,1))
    ax.set_yticks(range(0,6,1))
    ax.set_xticklabels(['x'+str(i) for i in range(len(Spearman))],fontsize=12)
    ax.set_yticklabels(['x'+str(i) for i in range(len(Spearman))],fontsize=12)

#fig.set_size_inches(18.5, 10.5, forward=True)
    fig.savefig('Room'+str(i)+'_Heatmap.png',dpi = 180)