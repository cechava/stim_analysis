#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:32:09 2019

@author: cesar
"""

# -*- coding: utf-8 -*-
import h5py

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import sys
import shutil
import glob
import optparse
import os
import json
import pandas as pd
import numpy as np
import pylab as pl
import scipy.stats as stats


in_filepath = './All_Images_layer0_activity_all.h5'
data_Grp = h5py.File(in_filepath, 'r')

original_L0_activ = np.array(data_Grp['layer_activity'])[1:56,:,:]
med_cmpx_L0_activ = np.array(data_Grp['layer_activity'])[55:111,:,:]
low_cmpx_L0_activ = np.array(data_Grp['layer_activity'])[110:,:,:]
data_Grp.close()

original_mean_all = np.mean(original_L0_activ.flatten())
med_cmpx_mean_all = np.mean(med_cmpx_L0_activ.flatten())
low_cmpx_mean_all = np.mean(low_cmpx_L0_activ.flatten())

print(original_mean_all,med_cmpx_mean_all,low_cmpx_mean_all)

original_mean_img = np.zeros((55,))
med_cmpx_mean_img = np.zeros((55,))
low_cmpx_mean_img = np.zeros((55,))

for img_idx in range(55):
    original_mean_img[img_idx] = np.mean(original_L0_activ[img_idx,:,:].flatten())
    med_cmpx_mean_img[img_idx] = np.mean(med_cmpx_L0_activ[img_idx,:,:].flatten())
    low_cmpx_mean_img[img_idx] = np.mean(low_cmpx_L0_activ[img_idx,:,:].flatten())
    
activity_df = pd.DataFrame({'img_mean': original_mean_img,
             'condition': 'original' })
activity_df =  activity_df.append(pd.DataFrame({'img_mean': med_cmpx_mean_img,
             'condition': 'medium cmpx' }))
activity_df = activity_df.append(pd.DataFrame({'img_mean': low_cmpx_mean_img,
             'condition': 'low cmpx' }))


palette = sns.color_palette(["#4c72b0","#c44e52","#55a868",])
sns.set_palette(palette)

p = sns.catplot(x='condition', y='img_mean', kind="swarm", hue = 'condition',data=activity_df,size = 10);

g = sns.jointplot(original_mean_img,med_cmpx_mean_img)
g.annotate(stats.pearsonr)
g = sns.jointplot(original_mean_img,low_cmpx_mean_img)
g.annotate(stats.pearsonr)
g = sns.jointplot(med_cmpx_mean_img,low_cmpx_mean_img)
g.annotate(stats.pearsonr)


data_Grp = h5py.File(in_filepath, 'r')

for img_idx in range(165):
    img_activity = np.array(data_Grp['layer_activity'])[img_idx,:,:].flatten()
    if img_idx == 0:
        L0_activ_all = img_activity
    else:
        L0_activ_all = np.vstack((L0_activ_all, img_activity))
        
data_Grp.close()

L0_activ_all_zscore = stats.zscore(L0_activ_all,0)

#replace nans with 0's
L0_activ_all_zscore[np.isnan(L0_activ_all_zscore)] = 0

#get correlations
R_all_zscore = np.corrcoef(L0_activ_all_zscore)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_all_zscore,center = 0,annot=False, cmap = 'RdBu_r')
