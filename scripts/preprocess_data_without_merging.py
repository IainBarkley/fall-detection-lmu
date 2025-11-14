import pandas as pd
import numpy as np
import sys,os

sensor_data_dir = '../fall_detection_data/sensor_data'
out_dir = '../fall_detection_data/not_merged_processed'

subjects = os.listdir(sensor_data_dir)

for subject in subjects:
    print('loading labelled data for subject {}'.format(subject))

    # load the labelled data
    labelled_data_dir = '../fall_detection_data/label_data'
    subject_filename = '{}_label.xlsx'.format(subject)
    lbdf = pd.read_excel(os.path.join(os.path.dirname(__file__),labelled_data_dir,subject_filename))
    lbdf['Task Code (Task ID)'] = lbdf['Task Code (Task ID)'].ffill()
    lbdf['Description'] = lbdf['Description'].ffill()
   
    lbdf['Subject Code'] = subject
    lbdf['Task Code'] = 'T' + lbdf['Task Code (Task ID)'].str.split(' ',1,expand=True)[1].str.strip('()')
    lbdf['Trial Code'] = 'R0' + lbdf['Trial ID'].astype(str)
    lbdf['Sensor Data Code'] = lbdf['Subject Code'] + lbdf['Task Code'] + lbdf['Trial Code']

    # load the sensor data
    trial_files = [f for f in os.listdir(os.path.join(sensor_data_dir,subject)) if '.csv' in f]
    
    for i,trial_file in enumerate(trial_files):
        temp_df = pd.read_csv(os.path.join(sensor_data_dir,subject,trial_file))
        
        # create a new dataframe for the individual if this is the first sensor data file we're grabbing
        ivdf = pd.DataFrame(columns = temp_df.columns)
            
        # grab the corresponding label data
        sensor_data_code = trial_file[:-4]
        sensor_data_code = sensor_data_code[:1] + 'A' + sensor_data_code[1:]
        trial_info = lbdf[ lbdf['Sensor Data Code'] == sensor_data_code ].reset_index(drop=True)
        
        if trial_info.shape[0] > 0:
            fall_onset_frame = trial_info['Fall_onset_frame'].iloc[0]
            fall_impact_frame = trial_info['Fall_impact_frame'].iloc[0]
            temp_df['Fall/No Fall'] = np.where( (temp_df['FrameCounter'] > fall_onset_frame) & (temp_df['FrameCounter'] < fall_impact_frame), 1, 0)
            temp_df['Fall Type'] = np.where( (temp_df['FrameCounter'] > fall_onset_frame) & (temp_df['FrameCounter'] < fall_impact_frame), trial_info['Task Code'], 0)
        
        else:
            temp_df['Fall/No Fall'] = 0
            temp_df['Fall Type'] = 0
        
        temp_df['Sensor Data Code'] = sensor_data_code
        
        temp_df['Fall/No Fall'] = temp_df['Fall/No Fall'].astype(int)
        temp_df['Fall Type'] = temp_df['Fall Type']
        
        out_filename = '{}-merged.csv'.format(sensor_data_code)
        temp_df.to_csv(os.path.join(out_dir,out_filename))
