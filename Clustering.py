import pandas as pd
import numpy as np
from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from keras.optimizers import Adagrad


def cluster(timeseries_df,data_labels):
    Shapelet_list = []
    D = '/home/abhilash/Datasets/UCRArchive_2018/TwoLeadECG/TXT_Files/'
    for i in range(1,max(data_labels)+1):
        print('Class',i)
        ts_df=timeseries_df[timeseries_df['0']==i]
        ts_df=ts_df.reset_index(drop=True)
        labels = ts_df['0']
        ts_df = ts_df.drop(ts_df.columns[0], axis=1)
        S='class'+str(i)+'labels.txt'
        pred_label=pd.read_csv(D+S,header=None)
        pred_label=np.ravel(np.array(pred_label))

        shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=ts_df.shape[0],
                                                               ts_sz=ts_df.shape[1],
                                                               n_classes=2,
                                                               l=0.36,
                                                               r=1)
        shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                            optimizer=Adagrad(lr=.1),
                            weight_regularizer=.01,
                            max_iter=50,
                            verbose=0)
        shp_clf.fit(ts_df, pred_label)
        shapelets=shp_clf.shapelets_;
        temp_list=[]
        for i in range(0, shapelets.shape[0]):
            temp= shapelets[i].T
            temp_list.append(temp[0])
        Shapelet_list.append(temp_list)
    return Shapelet_list
