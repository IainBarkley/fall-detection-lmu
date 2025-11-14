# vis_fall_durations.py

import matplotlib.pyplot as plt
import pandas as pd
import sys,os

training_data_dir = '../fall_detection_data/not_merged_processed'

fall_durations = []
for data_file in os.listdir(training_data_dir):
    ddf = pd.read_csv(os.path.join(training_data_dir,data_file),index_col=0)
    SR = 1 / ddf['TimeStamp(s)'].diff().mean()
    
    fall_duration = ddf['Fall/No Fall'].sum() / SR
    if fall_duration > 0.:
        fall_durations.append(fall_duration)

short = [ d for d in fall_durations if d < 1. ]
long = [ d for d in fall_durations if d > 1. ]

fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.hist(short,range = (0.,2.), bins = 40,rwidth=0.8,color='dimgray')
ax.hist(long,range = (0.,2.), bins = 40,rwidth=0.8,color='firebrick')
ax.axvline(1.,color='k',linestyle='--')
ax.set_ylabel('Fall Count')
ax.set_xlabel('Fall Duration (s)')

fig.tight_layout()
plt.show()

samples_included = sum( [ d*SR for d in short ] )
samples_excluded = sum( [ d*SR for d in long ] )

    
qty = [samples_included,samples_excluded]
fig,ax= plt.subplots(1,1,figsize=(3,3))
n,_,autotexts = ax.pie(qty,colors=['dimgray','firebrick'],autopct='%1.1f%%',pctdistance=0.6)
for autotext in autotexts:
    autotext.set_color('white')
    
plt.show()