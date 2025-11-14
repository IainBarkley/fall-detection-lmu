import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_scattermeans(df,features,ax,ylabel = None,colors = None,xlabels = None,min=0.0,max=1.):

    x = [x for x in range(1,len(features)+1)]

    w = 0.8
    for i,feature in enumerate(features):
        y = df[feature]
        print(feature,y.mean())
        
        # distribute scatter randomly across whole width of bar
        ax.scatter(x[i] + np.random.random(len(y)) * w / 2 - w / 4, y, color=colors[i],edgecolors='k',alpha = 0.8)
        ax.plot([x[i] - w / 4, x[i] + w / 4], [y.mean(),y.mean()], color='k',linewidth=2.0)
        ax.bar(x[i],
            height = y.mean(),
            yerr = y.std(),        # error bars
            capsize = 6,           # error bar cap width in points
            width = w,             # bar width
            color = (0,0,0,0),     # face color transparent
            edgecolor = None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
