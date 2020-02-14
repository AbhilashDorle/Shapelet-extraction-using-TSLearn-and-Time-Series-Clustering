import pandas as pd
import numpy as np
def generatefiles(timeseries_df,data_labels):
    for i in range(1,max(data_labels)+1):
        print('Class',i)
        ts_df=timeseries_df[timeseries_df['0']==i]
        ts_df=ts_df.reset_index(drop=True)
        ts_df=ts_df.to_numpy()
        filename='class'+str(i)+'.txt'
        np.savetxt('/home/abhilash/Datasets/UCRArchive_2018/CBF/TXT_Files/'+filename,ts_df)

timeseries_df=pd.read_csv(r"/home/abhilash/Datasets/UCRArchive_2018/CBF/CBF_TRAIN.tsv",sep="\t")# Line 2 in table 2
labels=np.array(timeseries_df['0'])
generatefiles(timeseries_df,labels)