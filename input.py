import pandas as pd
import numpy as np
import uShapeletClustering.Clustering
import scipy.io
import DTW
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
timeseries_df=pd.read_csv(r"/home/abhilash/Datasets/UCRArchive_2018/TwoLeadECG/TwoLeadECG_TRAIN.tsv",sep="\t")# Line 2 in table 2
labels=np.array(timeseries_df['0'])
# num_classes=[str(x+1) for x in range(len(np.unique(labels)))]
num_classes=['one','two']
shapelets=uShapeletClustering.Clustering.cluster(timeseries_df,labels)

print(shapelets[0])
#INSERTING A [0}*n ARRAY IF THE NUMBER OF SHAPELETS FOR ALL CLASSES ARE NOT EQUAL
max_shapelets_num=0
for i in range(len(shapelets)):
    if len(shapelets[i])>max_shapelets_num:
        max_shapelets_num=len(shapelets[i])


for i in range(len(shapelets)):
    if len(shapelets[i])<max_shapelets_num:
        shapelets[i].append(np.array([0]*len(shapelets[0][0])))

print('The number of shapelets are (templatenum)',len(shapelets[0]))
print('The length of the shapelets is ',len(shapelets[0][0]))

#GENERATING A .mat FILE TO BE READ IN MATLAB
S=[]

for slist in shapelets:
    temp=np.zeros((len(slist),),dtype=np.object)
    temp[:]=slist
    S.append(temp)

matlab_dict={}
count=0
for i in range(len(shapelets)):
    matlab_dict[num_classes[count]]=S[i]
    count=count+1
# scipy.io.savemat('TwoLeadECG30.mat', matlab_dict)

for i in range(len(shapelets)):
    print(len(shapelets[i]))

DTW.d(shapelets)