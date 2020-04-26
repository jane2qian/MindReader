import warnings
import sys 
from os.path import expanduser,dirname, join   
import nibabel as nib
import numpy as np
import time
import nilearn;from nilearn import plotting
import brainiak;from brainiak.searchlight.searchlight import Searchlight
from brainiak import io
from brainiak.utils.utils import array_correlation
from brainiak.fcma.util import compute_correlation
from brainiak.funcalign.srm import SRM
from pathlib import Path
from shutil import copyfile
import h5py
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from scipy import stats;from scipy.stats import norm, pearsonr, zscore
from scipy.spatial.distance import squareform
import scipy.spatial.distance as sp_distance
from nilearn import datasets,image,masking
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import matplotlib.pyplot as plt
import seaborn as sns 
from mpi4py import MPI

np.set_printoptions(precision=2, suppress=True)
sns.set(style = 'white', context='poster', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
fdir = dirname('/data/neuro/LLS_audio/derivatives/FUNCS/')
# Make directory to save results
output_dir = expanduser(fdir + '/searchlight_results')
#PL=[[1,2,4,8,11,16,19,20,23,28,37,38,39,42,44,46,48],[3,6,7,14,21,24,26,29,32,35,40,41,43,45,47],[5,9,10,12,13,15,17,22,25,27,30,31,33,34,36]]
PL=[[1,2,4,8,11,16,19,20,23,28,37,38,39,42,44,46,48]]
#group=dict([('H',PL[0]),('M',PL[1]),('L',PL[2])])
group=dict([('H',PL[0])])
mask_name = join(fdir,'results','GM_47_avg_mask.nii.gz')
GM_mask = nib.load(mask_name)
GM_mask = GM_mask.get_data()
ot_array=[]
dset =[]
gg=['H']
def load_4D_data():
    #subjects = [range(1,29),range(30,49)]
    subjects = PL
    for subject in subjects:
        for sub in subject:
            single_func=[]
            for run in range(1,5):
                bold = nib.load(join(fdir, 'sub{:03d}.run{:02d}.func.nii.gz').format(sub,run)) 
                affine_mat = bold.affine
                dimsize=bold.header.get_zooms()
                bold = bold.get_data()
                single_func.append(bold)
            single_func=np.concatenate(single_func[:],axis=3)
            dset.append(single_func)
    return dset,dimsize,affine_mat

#labels = ['H']*len(PL[0]) + ['M']*len(PL[1]) + ['L']*len(PL[2])
# set searchlight parameters
#mask = np.zeros(GM_mask.shape)
#mask[50:61,30:41,40:51] = 1
mask = GM_mask
sl_rad = 1
max_blx_edge = 5
pool_size = 1
bcvar=None
nfeature = 10;n_iter=10
if nfeature > (1+2*sl_rad)**3:
    print ('nfeature truncated')
    nfeature = int((1+2*sl_rad)**3)

def generate_train_dset(sl_data,sl_msk):
    data=[]
    for func in sl_data:
        dimx = sl_msk.shape[0];dimy=sl_msk.shape[1];dimz=sl_msk.shape[2];dimt = func.shape[3]
        data.append(np.reshape(func,(dimx*dimy*dimz,dimt)))
    data=np.array(data)
    data = stats.zscore(data,axis=1,ddof=1) 
    data = np.nan_to_num(data)  
    num_subs, vox_num, nTR = data.shape
    print('searchligh mask shape : ',sl_msk.shape)
    print( 'data in searchlight: ',
            'participants number ',num_subs,
            'Voxels per participant ', vox_num,
            'TRs per participant ', nTR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    return data
# set up predictSRM in searchlight kernel 
def srModel(sub,RankCorr,subj,data,name,bcast_var,g):
    print('fit srm in group {}'.format(name))
    train_dset = np.zeros((len(subj),data[0].shape[0],data[0].shape[1]))
    for i in range(len(subj)):
        j = subj[i]
        if j < 30:
            train_dset[i] = data[j-1] 
        else:
            train_dset[i] = data[j-2]
    subj, voxels, TRs = train_dset.shape
    print('subject number =', subj,
            'voxel number =',voxels,
            'times steps =', TRs)
    if sub < 30:
        original_signal = data[sub-1]
    else:
        original_signal = data[sub-2]
    srm = SRM(n_iter=bcast_var[0], features=bcast_var[1])
    print('srm fit train dataset',
            'feature number =', bcast_var[1],
            'iteration =', bcast_var[0])
    srm.fit(train_dset)
    shared_signal = srm.s_
    print('transforming new subject sub{:03d}'.format(sub))
    new_w = srm.transform_subject(original_signal)
    recons_signal = new_w.dot(shared_signal)
    print('DONE: reconstructed signal of new subject')
    recons_signal = stats.zscore(recons_signal)
    recons_signal = np.nan_to_num(recons_signal)
    print('compute correlation between original signal and reconstructed signal...')
    corr = array_correlation(recons_signal[...],original_signal[...],axis=1)[np.newaxis,:]
    meancorr = np.mean(corr)
    RankCorr[g][name].append(meancorr)
    return RankCorr

def validation_group(g,data,group,bcast_var,RankCorr,subjects):
    for name, train_sub in group.items():
        RankCorr[g][name]=[]
        for sub in subjects:
            if name == g:
                subj = train_sub[:]
                subj.remove(sub)
            if name != g:
                subj = train_sub[:]
            srModel(sub,RankCorr,subj,data,name,bcast_var,g)
        # RankOrder = sorted(RankCorr,key=RankCorr.get)
        # BestModel = RankOrder[2]
        # SecBestModel = RankOrder[1]
        # return BestModel,SecBestModel
def PredictSRM(sl_data,sl_msk,sl_rad,bcast_var):
    t1 = time.time()
    data=generate_train_dset(sl_data,sl_msk)
    RankCorr = dict()
    for g in gg:
        subjects = group[g][:]
        RankCorr[g]={}
        validation_group(g,data,group,bcast_var,RankCorr,subjects)
        AccCorr = RankCorr[g]
        print('prediction results for group {}'.format(g), AccCorr)
        for r in range(len(AccCorr[g])):
            acc=[]
            values = list(AccCorr.values())
            MaxCorr = max(values[0][r],values[1][r],values[2][r])
            if AccCorr[g][r]== MaxCorr:
                acc.append(1)
        acc=sum(acc)/len(AccCorr[g])   
        print('prediction accuracy for group {} is '.format(g),acc)   
        return acc  
    t2 = time.time()
    print('kernel duration: %.2f\n\n' % (t2-t1))

begin_time = time.time()
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blx_edge)
if rank==0:
    dset,dimsize,affine_mat=load_4D_data()
sl.distribute(dset,mask)
sl.broadcast([n_iter,nfeature])
print("Begin Searchlight\n")
sl_result = sl.run_searchlight(PredictSRM, pool_size=pool_size)
print("End Searchlight\n")
end_time = time.time()
# Print outputs
print("Summarize searchlight results")
print("Number of searchlights run: " + str(len(sl_result[mask==1])))
print("Accuracy for each kernel function: " +str(sl_result[mask==1].astype('double')))
print('Total searchlight duration (including start up time): %.2f' % (end_time - begin_time))
# Save the results to a .nii file
output_name = join(output_dir, '{}G_SL_result.nii.gz'.format(gg))
sl_result = sl_result.astype('double')  
sl_result[np.isnan(sl_result)] = 0  
sl_nii = nib.Nifti1Image(sl_result, affine_mat) 
hdr = sl_nii.header 
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
nib.save(sl_nii, output_name)  
