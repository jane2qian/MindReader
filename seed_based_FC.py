import nibabel as nib
import numpy as np
import os 
from os.path import dirname, join   
import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
import brainiak.utils.fmrisim as sim
from brainiak.fcma.util import compute_correlation
from nilearn import input_data
import time
from statsmodels.stats.multitest import multipletests,fdrcorrection
from scipy.stats import pearsonr,spearmanr
from nilearn import plotting
#seed corr functon
#seeds = [2,3,4,8,10,13,21,24,43,44]
seeds =[1,5,28,29,33,34]
def seed_correlation(wbBold, seedBold):
    """Compute the correlation between a seed voxel vs. other voxels 
    Parameters
    ----------
    wbBold [2d array], n_stimuli x n_voxels 
    seedBold, 2d array, n_stimuli x 1

    Return
    ----------    
    seed_corr [2d array], n_stimuli x 1
    seed_corr_fishZ [2d array], n_stimuli x 1
    """
    num_voxels = wbBold.shape[1]
    seed_corr = np.zeros((num_voxels, 1))
    for v in range(num_voxels):    
        seed_corr[v, 0] = np.corrcoef(seedBold.flatten(), wbBold[:, v])[0, 1]
    # Transfrom the correlation values to Fisher z-scores    
    seed_corr_fishZ = np.arctanh(seed_corr)
    return seed_corr, seed_corr_fishZ
fdir = dirname('/data/neuro/LLS_audio/derivatives/FUNCS/')
PL=[[1,2,4,8,11,16,19,20,23,28,38,39,42,44,48],[3,6,7,14,21,24,32,35,37,40,41,43,45,46,47],[5,9,10,12,13,15,17,18,22,25,26,27,30,31,33,34,36]]
labels=['H','M','L']
atlas=datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-2mm',data_dir='/usr/local/')
atlas_filename=atlas.maps
atlas_pd=pd.DataFrame(atlas)
masker_ho=NiftiLabelsMasker(labels_img=atlas_filename)
masker_wb = input_data.NiftiMasker(
    standardize=True,  # Are you going to zscore the data across time?
    t_r=1.5, 
    memory='nilearn_cache',  # Caches the mask in the directory given as a string here so that it is easier to load and retrieve
    memory_level=1,  # How much memory will you cache?
    verbose=0)

### aal for musicsynatx plotting
bold=nib.load('/Users/ruiqing/Desktop/MusicSyntax/MUSIC/101/BOLD/swachenxue_20161107_001_007_ep2d_run3.nii')
aff=bold.affine
atlas=datasets.fetch_atlas_aal(version='SPM12',data_dir='/Applications/matlab_toolbox/')
atlas_filename=atlas.maps
atlas_pd=pd.DataFrame(atlas)
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache')
data = np.ones((53, 63, 46,1))
data2 = np.zeros((53, 63, 46,1))
img = nib.Nifti1Image(data,affine=aff)
img2 =nib.Nifti1Image(data2,affine=aff)
indx = [15,74]
coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
plotting.plot_connectome(img, coordinates,
                         edge_threshold="80%")
from nilearn import plotting, image
from nilearn.image import concat_imgs, index_img
joint_mni_image = concat_imgs([datasets.fetch_atlas_aal(version='SPM12',data_dir='/Applications/matlab_toolbox/'),
 datasets.fetch_atlas_aal(version='SPM12',data_dir='/Applications/matlab_toolbox/')])
# First plot the map for the PCC: index 4 in the atlas
display = plotting.plot_stat_map(image.index_img(atlas_filename, 15),colorbar=False,title="Frontal_Inf_Orb_R~Pallidum_L")
display.add_overlay(image.index_img(atlas_filename, 74),
                    cmap=plotting.cm.black_pink)
nodes = image.index_img(atlas_filename, [15,74])
# Note that dmn_node is now a 4D image
print(nodes.shape)
display = plotting.plot_prob_atlas(nodes,title="Frontal_Inf_Orb_R~Pallidum_L")
plotting.show()
# (seed_id +1):                                   regions:
# 1                                          Frontal Pole
# 2                                        Insular Cortex
# 3                                Superior Frontal Gyrus
# 4                                  Middle Frontal Gyrus
# 5             Inferior Frontal Gyrus, pars triangularis
# 6              Inferior Frontal Gyrus, pars opercularis
# 7                                      Precentral Gyrus
# 8                                         Temporal Pole
# 9            Superior Temporal Gyrus, anterior division
# 10          Superior Temporal Gyrus, posterior division
# 11             Middle Temporal Gyrus, anterior division
# 12            Middle Temporal Gyrus, posterior division
# 13         Middle Temporal Gyrus, temporooccipital part
# 14           Inferior Temporal Gyrus, anterior division
# 15          Inferior Temporal Gyrus, posterior division
# 16       Inferior Temporal Gyrus, temporooccipital part
# 17                                    Postcentral Gyrus
# 18                             Superior Parietal Lobule
# 19               Supramarginal Gyrus, anterior division
# 20              Supramarginal Gyrus, posterior division
# 21                                        Angular Gyrus
# 22          Lateral Occipital Cortex, superior division
# 23          Lateral Occipital Cortex, inferior division
# 24                                Intracalcarine Cortex
# 25                                Frontal Medial Cortex
# 26    Juxtapositional Lobule Cortex (formerly Supple...
# 27                                   Subcallosal Cortex
# 28                                  Paracingulate Gyrus
# 29                   Cingulate Gyrus, anterior division
# 30                  Cingulate Gyrus, posterior division
# 31                                    Precuneous Cortex
# 32                                        Cuneal Cortex
# 33                               Frontal Orbital Cortex
# 34             Parahippocampal Gyrus, anterior division
# 35            Parahippocampal Gyrus, posterior division
# 36                                        Lingual Gyrus
# 37          Temporal Fusiform Cortex, anterior division
# 38         Temporal Fusiform Cortex, posterior division
# 39                   Temporal Occipital Fusiform Cortex
# 40                             Occipital Fusiform Gyrus
# 41                             Frontal Operculum Cortex
# 42                             Central Opercular Cortex
# 43                            Parietal Operculum Cortex
# 44                                        Planum Polare
# 45                  Heschl's Gyrus (includes H1 and H2)
# 46                                     Planum Temporale
# 47                                Supracalcarine Cortex
# 48                                       Occipital Pole

def load_single_4D_data():
    print('loading subject{:03d} bold file...'.format(sub))
    single_func=[]
    for run in range(1,5):
        bold = nib.load(join(fdir,'sub{:03d}.run{:02d}.func.nii.gz').format(sub,run)) 
        aff=bold.affine
        bold = bold.get_data()
        single_func.append(bold)
    single_func=np.concatenate(single_func[:],axis=3)
    return single_func,aff
for seed_id in seeds:    
    print('seed {} FC computing'.format(seed_id))
    for gg in range(0,3):
        g=labels[gg]
        subjects=PL[gg]
        matls = []
        matls_z=[]
        for sub in subjects:
            dset,aff=load_single_4D_data()
            dset=nib.Nifti1Image(dset,affine=aff)
            bold_ho=masker_ho.fit_transform(dset)
            bold_wb=masker_wb.fit_transform(dset)
            seed_ho = np.array(bold_ho[:,seed_id])
            seed_ho = seed_ho.reshape(seed_ho.shape[0],-1)
            corr_mat_ho,corr_mat_ho_z = seed_correlation(bold_wb,seed_ho)
            #correlation_measure = ConnectivityMeasure(kind='correlation')
            #corr_mat_ho = correlation_measure.fit_transform([bold_ho])[0]
            matls.append(corr_mat_ho)
            group_corr_mat_ho = np.stack(matls)
            matls_z.append(corr_mat_ho_z)
            group_corr_mat_ho_z = np.stack(matls_z)
            img_corr_seed=masker_wb.inverse_transform(corr_mat_ho_z.T)
            nib.save(img_corr_seed,'seed_{}_{}_sub{:03d}.nii.gz'.format(seed_id+1,g,sub))
            print('group corrlation dimension is ',group_corr_mat_ho.shape)
        np.save(os.path.join(fdir,'FC','seed_{}_{}.npy'.format(seed_id+1,g)),group_corr_mat_ho)
        np.save(os.path.join(fdir,'FC','seed_{}_{}_fisherz.npy'.format(seed_id+1,g)),group_corr_mat_ho_z) 


subjects=[1,2,4,8,11,16,19,20,23,28,38,39,42,44,48,3,6,7,14,21,24,32,35,37,40,41,43,45,46,47,5,9,10,12,13,15,17,18,22,25,26,27,30,31,33,34,36]
region =['IFG,pars triangularis','IFG,pars opercularis','Temporal Pole',
        'MTG,anterior','MTG,posterior','ITG,anterior','Medial Frontal gyrus','Cingulate Gyrus,anterior','Cingulate Gyrus,posterior',
        'precuneus','Parahippocampal,anterior','Parahippocampal,posterior','Planum Polare',"Heschl's gyrus",'Planum Temporale']
seed=[5,6,8,11,12,14,25,29,30,31,34,35,44,45,46]
  
# img_corr_seed=masker_wb.inverse_transform(corr_mat_ho_z.T)
for i in range(len(seed)):
    s = seed[i]
    alpha=0.05
#[1,5,28,29,33,34,2,3,4,8,10,13,21,24,43,44]:
    # for gg in range(0,3):
    #     g=labels[gg]
    #     subjects=PL[gg]
    rtank = []
    for sub in subjects[15:30]:
        bold = nib.load('seed{}_H_sub{:03d}.nii.gz'.format(s,sub))
        aff=bold.affine
        bold = bold.get_data()
        # corr_fisherz = np.arctanh(bold)
        # corr_fisherz = np.nan_to_num(corr_fisherz)
        # rtank.append(corr_fisherz)
        rtank.append(bold)
    rtank = np.array(rtank)
    rtank = rtank.reshape((len(subjects[15:30]),91,109,91))
    avg = np.average(rtank,axis=0)
    rtank1 = []
    for sub1 in subjects[30:47]:
        bold1 = nib.load('seed{}_H_sub{:03d}.nii.gz'.format(s,sub1))
        bold1 = bold1.get_data()
        # corr_fisherz = np.arctanh(bold)
        # corr_fisherz = np.nan_to_num(corr_fisherz)
        # rtank.append(corr_fisherz)
        rtank1.append(bold1)
    rtank1 = np.array(rtank1)
    rtank1 = rtank1.reshape((len(subjects[30:47]),91,109,91))
    avg1 = np.average(rtank1,axis=0)
    t,p = stats.ttest_ind(rtank,rtank1)#popmean=0,axis=0)
    p = np.ravel(p)
    t=np.ravel(t)
    #rej,pval_corr=fdrcorrection(p)
    rej,pval_corr,alpsidak,alpbonf=multipletests(p,alpha=alpha,method='bonferroni')
    if not pval_corr[pval_corr<alpha].size:
        print('no significant difference under FWER 0.05 ') 
    else:
        sigt=t[pval_corr<alpha]
        thres=min(abs(sigt))
        t=np.reshape(t,(91,109,91))
        img=nib.Nifti1Image(t,affine=aff)
        nib.save( img,'{}_HvsL'.format(region[i]))
        plotting.plot_glass_brain(img,threshold=thres,colorbar=True,display_mode='lyrz',plot_abs=False,
                                    output_file='seed{}_{}_HvsL.png'.format(s,region[i]),
                                    title='{}'.format(region[i]))
        
#region = ['Superior Frontal Gyrus','Middle Frontal Gyrus','Inferior Frontal Gyrus, pars triangularis','Superior Temporal Gyrus',
 #           'Middle Temporal Gyrus','Inferior Temporal Gyrus','Lateral Occipital Cortex','Frontal Medial Cortex','Planum Polare', "Heschl's Gyrus"]
#s = [2,3,4,8,10,13,21,24,43,44]
# plotting randomise results
#'Insula']#,'Inferior Frontal Gyrus, pars opercularis','Cingulate Gyrus(anterior)','Cingulate Gyrus(posterior)',
          #  'Parahippocampal Gyrus(anterior)','Parahippocampal Gyrus(posterior)']
s = [43,44]
for i in range(len(region)):
    ss = s[i]
    name = region[i]
    for n in ['HM']:
        stats_imgs = nib.load('/Users/ruiqing/Desktop/FC/seed_FC_randomise_group/seed{}_{}_tfce_corrp_tstat1.nii.gz'.format(ss+1,n))
        display = plotting.plot_glass_brain(None,display_mode='lr',colorbar=True)
        display.add_contours(stats_imgs,levels=[0.95],colors='r')
        display.title('{}:{}'.format(n[0],name)) 
        display.savefig('{}_{}.png'.format(n[0],name))
       #plotting.plot_glass_brain(stats_imgs,threshold=0.95,colorbar=True,display_mode='lzr',output_file='{}_{}.png'.format(n[0],name))
       
        #plotting.plot_glass_brain(stats_imgs,threshold=0.95,colorbar=True,display_mode='lzr',plot_abs=False,title='{}:{}'.format(name,n[0]),
        #                            output_file='{}_{}.png'.format(n[0],name))

g=['H','L']
stats_imgs = nib.load('/Users/ruiqing/Desktop/searchlight_PredictSRM/group_comp_3/{}vs{}_acc_tfce_corrp_tstat1.nii'.format(g[0],g[1]))
display = plotting.plot_glass_brain(None,display_mode='lzr',colorbar=True)
display.add_contours(stats_imgs,colors='r')
display.title('{} SRM prediction accuracy'.format(g[1])) 
display.savefig('{}.png'.format(g[1]))
# multi-regress on FC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from sklearn.cross_decomposition import PLSRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
comp_score =  [5.5,6.25,5.5,5.75,5.5,5.25,5.5,5.25,5.75,5.75,5,5.5,6.5,5.75,6.25,5.5,5.25,
                4.5,4.75,4.75,4.75,3.75,4.25,4,3.75,4,4.5,4.75,4.75,4.75,3.75,4.75,3.5,
                3.25,3,2.75,3,3,0.5,3.5,2.5,1.25,0.75,3.25,2.25,3.5,3.25]
# phon_score = [0.780487805,None,0.829268293,0.756097561,0.609756098,0.829268293,0.951219512,	0.609756098,0.804878049,1,
# 	            0.902439024,0.731707317,0.975609756,	0.731707317,	0.951219512,	0.658536585,	0.756097561,	0.658536585,
#                 	0.756097561,	0.731707317,	0.853658537, None,	0.707317073,	0.390243902,	0.756097561,	0.731707317,
#                     	0.390243902,	0.829268293,	0.87804878,	0.609756098,	0.804878049,	0.853658537,	0.829268293,	0.926829268,
#                         	0.780487805,	0.829268293,	0.756097561,	0.756097561,	0.804878049,	0.219512195,	0.731707317,	0.707317073	,
#                             0.902439024,	0.756097561,	0.804878049,	0.756097561,	0.658536585]
phon_score_z=[ 0.15656296,0.925,  0.49528089,  0.02107578, -0.99507802,  0.49528089,
        1.30820393, -0.99507802,  0.29205013,  1.64692186,  0.96948599,
       -0.18215498,  1.51143469, -0.18215498,  1.30820393, -0.65636008,
        0.02107578, -0.65636008,  0.02107578, -0.18215498,  0.63076806,0.65,
       -0.31764215, -2.48543692,  0.02107578, -0.18215498, -2.48543692,
        0.49528089,  0.83399882, -0.99507802,  0.29205013,  0.63076806,
        0.49528089,  1.17271675,  0.15656296,  0.49528089,  0.02107578,
        0.02107578,  0.29205013, -3.63707789, -0.18215498, -0.31764215,
        0.96948599,  0.02107578,  0.29205013,  0.02107578, -0.65636008]

years=[13,23,13,	20,	15,	12,	17,	22,	14,
	14,	18,	16,	15,	14,	12,	16,
	14,	7,	13,	14,	10,	7,	13,	15,	10,	10,	12,	13,
	11,	15,	13,	16,	15,	16,	11,	17,	13,
	17,	11,	15,	7,	12,	20,	11,	15,	14,	12]
# for i in range(len(phon_score)):
#     if phon_score[i] !=None:
#         phon_score[i]=round(phon_score[i],2)
# subj_list={'subjects':subjects}
# pd_sub = pd.DataFrame(subj_list)
comp_score_z=zscore(comp_score)
years_z=zscore(years)
# phon_score=zscore(phon_score)
years_z=zscore(years)
data = {'comp':comp_score_z[0:15],'phon':phon_score_z[0:15],'years':years_z[0:15]}
df = pd.DataFrame(data)
predict_signal=[]
#seed_id+1:Temporal Pole(8);Middle Temporal Gyrus, posterior division(12);Planum Temporale(46)
for seed_id in [6,7,9,11,12,16,30,38,40,45,2,3,4,8,10,13,21,24,43,44,1,5,28,29,33,34]:
#    for gg in range(0,3)
#         g=labels[gg]
        #subjects=PL[gg]
    #avg_corr=[]
    Y=[]
    avg_bold=[]
    for sub in subjects[0:15]:
        bold = nib.load('seed{}_H_sub{:03d}.nii.gz'.format(seed_id+1,sub))
        aff=bold.affine
        bold = bold.get_data()
        avg_bold = np.average(bold)
        # corr_fisherz = np.arctanh(bold)
        # corr_fisherz = np.nan_to_num(corr_fisherz)
        # avg_single_corr = np.average(corr_fisherz)
        # avg_corr.append(avg_single_corr)
        Y.append(avg_bold)
    Y=np.array(Y)
    #x = sm.add_constant(comp_score)
    model = smf.ols(formula='Y~comp',data=df)
    #model = smf.glm(formula='Y~comp*phon',data=df)
    reg = model.fit()
    reg.summary()
    # a,b=dmatrices('Y~phon',df,return_type='dataframe')
    # vif=pd.DataFrame()
    # vif['vif']=[variance_inflation_factor(b.values,i) for i in range(b.shape[1])]
    # vif['feature']=b.columns
    print('seed region ID:',seed_id+1)
    # print(vif)
    # pls=PLSRegression(n_components=2)
    # pls.fit(df,Y)
    # Y_predict=pls.predict(df)
    # predict_signal.append(Y_predict)
    # print('pls coef in seed {}:{}'.format(seed_id+1,pls.coef_))
  

