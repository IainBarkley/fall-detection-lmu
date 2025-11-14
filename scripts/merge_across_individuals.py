import pandas as pd
import numpy as np
import sys,os

merged_data_dir = '../fall_detection_data/merged_processed'
subject_files = os.listdir(merged_data_dir)

for s,subject_file in enumerate(subject_files):
    print('loading {}'.format(subject_file))
    
    with pd.read_csv(os.path.join(merged_data_dir,subject_file),index_col=0) as ddf:
    
    # check if train_df exists
    # is so: open it, add rows,
    
    # else, create it
    
    # do the same for test_df 
    
    print(ddf.head())
    
