# generate_fall_labels.py

import pandas as pd
import sys,os

labelled_data_dir = '../fall_detection_data/label_data'
subject = 'SA06'

subject_filename = '{}_label.xlsx'.format(subject)

lbdf = pd.read_excel(os.path.join(os.path.dirname(__file__),labelled_data_dir,subject_filename))
lbdf['Task Code (Task ID)'] = lbdf['Task Code (Task ID)'].ffill()
lbdf['Description'] = lbdf['Description'].ffill()
lbdf['Subject Code'] = subject
lbdf['Task Code'] = 'T' + lbdf['Task Code (Task ID)'].str.split(' ',1,expand=True)[1].str.strip('()')
lbdf['Trial Code'] = 'R0' + lbdf['Trial ID'].astype(str)
lbdf['Sensor Data Code'] = lbdf['Subject Code'] + lbdf['Task Code'] + lbdf['Trial Code']
print(lbdf.head())
