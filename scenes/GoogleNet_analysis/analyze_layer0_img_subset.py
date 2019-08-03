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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

img_subset = np.array([2,3,23,28,34,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55])

nimgs_per_class = 55

img_list = np.arange(0,20)
cond_list = np.arange(0,3)

#get activity
in_filepath = './All_Images_layer0_activity_all.h5'
data_Grp = h5py.File(in_filepath, 'r')
for img_count,img in enumerate(img_subset):
    #img = img_subset[0]-1
    img = img-1
    print(img)
    original_idx = img
    med_cmpx_idx = img+nimgs_per_class
    low_cmpx_idx = img+(nimgs_per_class*2)
    print(original_idx, med_cmpx_idx,low_cmpx_idx)
    
    original_activ = np.array(data_Grp['layer_activity'])[original_idx,:,:].flatten()
    med_cmpx_activ = np.array(data_Grp['layer_activity'])[med_cmpx_idx,:,:].flatten()
    low_cmpx_activ = np.array(data_Grp['layer_activity'])[low_cmpx_idx,:,:].flatten()
    activ_set = np.vstack((original_activ, med_cmpx_activ, low_cmpx_activ))
    
    if img_count == 0:
        all_activity = activ_set
    else:
        all_activity = np.vstack((all_activity,activ_set))
data_Grp.close()

nconfigs, num_units = all_activity.shape

#check some feature maps for corresponding images
img= 2
filter = 0

original_idx = (3*img)
med_cmpx_idx = (3*img)+1
low_cmpx_idx = (3*img)+2

feat_map_orig = np.swapaxes(np.reshape(all_activity[original_idx,:],(64,56,56)),1,2)
feat_map_med_cmpx = np.swapaxes(np.reshape(all_activity[med_cmpx_idx,:],(64,56,56)),1,2)
feat_map_low_cmpx = np.swapaxes(np.reshape(all_activity[low_cmpx_idx,:],(64,56,56)),1,2)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(feat_map_orig[filter,:,:],center = 0)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(feat_map_med_cmpx[filter,:,:],center = 0)


plt.figure(figsize=(12, 10))
ax = sns.heatmap(feat_map_low_cmpx[filter,:,:],center = 0)



#normalize by max across conditions
max_response = np.max(all_activity,0)
max_response = np.expand_dims(max_response,0)
max_response = np.tile(max_response,(60,1))
all_activity_norm = all_activity/max_response

#avg stim response across units
img_activ_mean = np.nanmean(all_activity_norm,1)
img_activ_se = np.nanstd(all_activity_norm,1)/all_activity_norm.shape[1]


#plot mean of each stim across units
bar_loc = np.zeros((img_activ_mean.size))
width = 0.4         # the width of the bars
gap = .5
xloc = 1
count = 0
for i in range(20):
    for j in range(3):
        bar_loc[count] = xloc
        xloc = xloc + width
        count = count+1
    xloc = xloc + gap


fig = plt.figure(figsize=(20,5))
plt.bar(bar_loc[0:len(bar_loc):3],img_activ_mean[0:len(bar_loc):3],width,color = 'b',yerr = img_activ_se[0:len(bar_loc):3])
plt.bar(bar_loc[1:len(bar_loc):3],img_activ_mean[1:len(bar_loc):3],width,color = 'g',yerr = img_activ_se[1:len(bar_loc):3])
plt.bar(bar_loc[2:len(bar_loc):3],img_activ_mean[2:len(bar_loc):3],width,color = 'r',yerr = img_activ_se[2:len(bar_loc):3])

axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = bar_loc[1:len(bar_loc):3]
xtick_label = np.unique(img_list+1).astype('int')

plt.xticks(xtick_loc,xtick_label.tolist())
plt.xlabel('Image',fontsize = 15)
plt.ylabel('Average Activity',fontsize = 15)


#fig_fn = 'avg_across_neurons_%s_thresh_%s_%i.png'%(response_type,filter_crit,filter_thresh)
#fig_file_path = os.path.join(fig_out_dir, fig_fn)
#plt.savefig(fig_file_path)
#plt.close()



#get and plot correlations across images
g = sns.jointplot(img_activ_mean[0:nconfigs:3],img_activ_mean[1:nconfigs:3])
g.annotate(stats.pearsonr)
plt.suptitle('Original vs Medium Complex - per image')

g = sns.jointplot(img_activ_mean[0:nconfigs:3],img_activ_mean[2:nconfigs:3])
g.annotate(stats.pearsonr)
plt.suptitle('Original vs Low Complex - per image')


g = sns.jointplot(img_activ_mean[1:nconfigs:3],img_activ_mean[2:nconfigs:3])
g.annotate(stats.pearsonr)
plt.suptitle('Medium vs Low Complex - per image')



 #average over units for each condition
activ_per_cond_per_unit = np.zeros((3,num_units))

for cond in range(3):
    tmp = all_activity_norm[cond:nconfigs:3,:]
    activ_per_cond_per_unit[cond,:] = np.nanmean(tmp,0)


activ_per_cond_mean_neuron = np.nanmean(activ_per_cond_per_unit,1)
activ_per_cond_se_neuron = np.nanstd(activ_per_cond_per_unit,1)/np.sqrt(num_units)

#plot
bar_loc = np.zeros((3,))
width = 0.4         # the width of the bars
xloc = 1
count = 0

for j in range(len(cond_list)):
    bar_loc[count] = xloc
    xloc = xloc + width
    count = count+1

fig = plt.figure(figsize=(8,8))
plt.bar(bar_loc[0],activ_per_cond_mean_neuron[0],width,color = 'b',yerr = activ_per_cond_se_neuron[0])
plt.bar(bar_loc[1],activ_per_cond_mean_neuron[1],width,color = 'g',yerr = activ_per_cond_se_neuron[1])
plt.bar(bar_loc[2],activ_per_cond_mean_neuron[2],width,color = 'r',yerr = activ_per_cond_se_neuron[2])

axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = []
xtick_label = []

plt.xticks(xtick_loc,xtick_label)
plt.xlabel('Condition',fontsize = 15)
plt.ylabel('Average Response',fontsize = 15)

#put things into pandas
activ_dfs = []
subset_units = np.random.randint(0,num_units,1000)
for cidx in cond_list:
    response = activ_per_cond_per_unit[cidx,subset_units]
    cell = np.arange(subset_units.size)
    cond = np.ones((subset_units.size,))*cidx
    mdf = pd.DataFrame({'response': response,
                        'cell': cell,
                        'cond': cond,
                       })

    activ_dfs.append(mdf)
activ_dfs = pd.concat(activ_dfs, axis=0)

#make swarm plot
bar_loc = np.arange(0,3)
width = 0.5

palette = sns.color_palette(["#4c72b0","#55a868","#c44e52"])
sns.set_palette(palette)

p = sns.catplot(x='cond', y='response', kind="swarm", hue = 'cond',data=activ_dfs,size = 10);

axes = p.ax
ymin,ymax = axes.get_ylim()
xmin,xmax = axes.get_xlim()

for idx in cond_list:
    p.ax.hlines(y = activ_per_cond_mean_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=2, color='k',linestyle = '-')
    p.ax.hlines(y = activ_per_cond_mean_neuron[idx] + activ_per_cond_se_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=1, color='k',linestyle = '--')
    p.ax.hlines(y = activ_per_cond_mean_neuron[idx] - activ_per_cond_se_neuron[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=1, color='k',linestyle = '--')



p.ax.set_xticks(())
p.ax.set_xlabel('Condition',fontsize = 15)
p.ax.set_ylabel('Response',fontsize = 15)
#p.fig.suptitle('Average %s Across Neurons'%(response_type),fontsize = 15)

sns.set_style('darkgrid')
#getting and plotting some correlations 
g = sns.jointplot(activ_per_cond_per_unit[0,:],activ_per_cond_per_unit[1,:])
g.annotate(stats.pearsonr)
plt.suptitle('Original vs Medium Complex - per unit')

#getting and plotting some correlations 
g = sns.jointplot(activ_per_cond_per_unit[0,:],activ_per_cond_per_unit[2,:])
g.annotate(stats.pearsonr)
plt.suptitle('Original vs Low Complex - per unit')

g = sns.jointplot(activ_per_cond_per_unit[1,:],activ_per_cond_per_unit[2,:])
g.annotate(stats.pearsonr)
plt.suptitle('Medium vs Low Complex - per unit')

#plotting modulation index

mod_idx = np.zeros((2,num_units))

#original > medium complexity modulation index
mod_idx[0,:] = np.true_divide(activ_per_cond_per_unit[0,:] - activ_per_cond_per_unit[1,:],activ_per_cond_per_unit[0,:] + activ_per_cond_per_unit[1,:])

#medium > low complexiity modulation index
mod_idx[1,:] = np.true_divide(activ_per_cond_per_unit[1,:] - activ_per_cond_per_unit[2,:],activ_per_cond_per_unit[1,:] + activ_per_cond_per_unit[2,:])

mod_idx_mean = np.nanmean(mod_idx,1)

#put things into pandas
mod_dfs = []
subset_units = np.random.randint(0,num_units,1000)
for cidx in range(2):
    mod = mod_idx[cidx,subset_units]
    cell = np.arange(subset_units.size)
    cond = np.ones((subset_units.size,))*cidx
    mdf = pd.DataFrame({'mod': mod,
                        'cell': cell,
                        'cond': cond,
                       })

    mod_dfs.append(mdf)
mod_dfs = pd.concat(mod_dfs, axis=0)

#make swarm plot
bar_loc = np.arange(0,3)
width = 0.5

palette = sns.color_palette('bright')
sns.set_palette(palette)

p = sns.catplot(x='cond', y='mod', kind="swarm", hue = 'cond',data=mod_dfs,size = 10);

axes = p.ax
ymin,ymax = axes.get_ylim()
xmin,xmax = axes.get_xlim()

for idx in range(2):
    p.ax.hlines(y = mod_idx_mean[idx], xmin=bar_loc[idx]-(width/2), xmax = bar_loc[idx]+(width/2), linewidth=2, color='k',linestyle = '-')



p.ax.set_xticks(())
p.ax.set_xlabel('Condition',fontsize = 15)
p.ax.set_ylabel('Modlation Index',fontsize = 15)
plt.legend(['Orig>Med','Med>Low'])

#p.fig.suptitle('Average %s Across Neurons'%(response_type),fontsize = 15)


#---RSA analysis----

#z-score across all images
all_activity_zscore = stats.zscore(all_activity,0)

#replace nans with 0's
all_activity_zscore[np.isnan(all_activity_zscore)] = 0

#get correlations
R_all_zscore = np.corrcoef(all_activity_zscore)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_all_zscore,center = 0,annot=False, cmap = 'RdBu_r')

#re-arrange activity to compare class
all_activity_zscore_class = np.vstack((all_activity_zscore[0:nconfigs:3,:],all_activity_zscore[1:nconfigs:3,:],all_activity_zscore[2:nconfigs:3,:]))
#get correlations
R_all_zscore_class = np.corrcoef(all_activity_zscore_class)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(R_all_zscore_class,center = 0,annot=False, cmap = 'RdBu_r')


#get correlational distance between each original image and its derivates
corr_distance = np.zeros((2,len(img_list)))
for img in img_list:
    corr_distance[0,img] = 1-R_all_zscore[img,img+1]#original to medium complex
    corr_distance[1,img] = 1-R_all_zscore[img,img+2]#original to low complexitiy


#plot 
bar_loc = np.zeros((len(img_list)*2))
width = 0.4         # the width of the bars
gap = .5
xloc = 1
count = 0
for i in range(20):
    for j in range(2):
        bar_loc[count] = xloc
        xloc = xloc + width
        count = count+1
    xloc = xloc + gap


fig = plt.figure(figsize=(20,5))
plt.bar(bar_loc[0:len(bar_loc):2],corr_distance[0,:],width,color = 'c')
plt.bar(bar_loc[1:len(bar_loc):2],corr_distance[1,:],width,color = 'm')

axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = bar_loc[1:len(bar_loc):2]
xtick_label = np.unique(img_list+1).astype('int')

plt.xticks(xtick_loc,xtick_label.tolist())
plt.xlabel('Image',fontsize = 15)
plt.ylabel('Correlational Distance',fontsize = 15)

#average and plot
bar_loc = np.arange(0,20)
fig = plt.figure(figsize=(20,5))
plt.bar(bar_loc,corr_distance[1,:]-corr_distance[0,:],width,color = 'b')

axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = bar_loc
xtick_label = np.unique(img_list+1).astype('int')

plt.xticks(xtick_loc,xtick_label.tolist())
plt.xlabel('Image',fontsize = 15)
plt.ylabel('Correlational Distance Difference',fontsize = 15)


#------DIMENSIONALITY ANALYSIS-------------

def get_pca_data(data,scaler,pca):
    scaler.fit(data)
    pca_data = scaler.transform(data)
    
    principal_components = pca.fit_transform(pca_data)
    exp_var =  pca.explained_variance_
    exp_var_ratio =  pca.explained_variance_ratio_
    return principal_components, exp_var, exp_var_ratio


#break off into corresponsding matrices
all_activity_original = all_activity[0:nconfigs:3,:]
all_activity_med_cmpx = all_activity[1:nconfigs:3,:]
all_activity_low_cmpx = all_activity[2:nconfigs:3,:]

#define some functions
pca = PCA()
scaler = StandardScaler()

#get PCA results
princ_comp_orig, exp_var_orig, exp_var_ratio_orig = get_pca_data(all_activity_original,scaler,pca)
princ_comp_med, exp_var_med, exp_var_ratio_med = get_pca_data(all_activity_med_cmpx,scaler,pca)
princ_comp_low, exp_var_low, exp_var_ratio_low = get_pca_data(all_activity_low_cmpx,scaler,pca)

#visualize cumulative explained variance ratio
sns.distplot(exp_var_ratio_orig, hist = False, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(exp_var_ratio_med,hist = False, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(exp_var_ratio_low,hist = False, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))

#get total variance and normalize
exp_var_sum_all = np.hstack((sum(exp_var_orig),sum(exp_var_med),sum(exp_var_low)))
exp_var_sum_all_norm = exp_var_sum_all/np.max(exp_var_sum_all)

#visualize total variance across stimulus sets
palette = sns.color_palette(["#4c72b0","#c44e52","#55a868"])
sns.set_palette(palette)
fig = plt.figure(figsize=(8,8))
sns.barplot(np.arange(0,3),exp_var_sum_all_norm)


axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.axhline(y=0, xmin=xmin, xmax= xmax, linewidth=1, color='k',linestyle = '-')

xtick_loc = []
xtick_label = []

plt.xticks(xtick_loc,xtick_label)
plt.xlabel('Condition',fontsize = 15)
plt.ylabel('Variance',fontsize = 15)

#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])


