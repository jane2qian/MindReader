# ISC anlysis: compute voxel-voxel between subjects
from os.path import abspath, dirname, join
from brainiak.isfc import isc, isfc
import numpy as np
import nibabel as nib
from brainiak import image, io
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
# put the script in the fc dir path
# set data path
root_dir = dirname(abspath("_file_"))
mask_file = join(root_dir,'practice_FC/isc','avg152T1_gray_3mm.nii.gz')
func_file = [join(root_dir,'practice_FC/isc','sub-{0:03d}-task-intact1.nii.gz'.format(sub))
            for sub in np.arange(1, 6)]
print('Loading data from {0} subjects...'.format(len(func_file)))

# load mask and apply mask to func data for all subjects
mask_bool = io.load_boolean_mask(mask_file, lambda x: x>50)
masked_func = image.mask_images(io.load_images(func_file), mask_bool)
coords = np.where(mask_bool)
all_masked_func = image.MaskedMultiSubjectData.from_masked_images(masked_func, len(func_file))

# compute isc 
isc_r = isc(all_masked_func)
isc_r = np.nan_to_num(isc_r)
# isc data to nii
nii_template = nib.load(mask_file)
isc_map = np.zeros(nii_template.shape)
isc_map[coords]=isc_r
isc_img = nib.Nifti1Image(isc_map, nii_template.affine, nii_template.header)
nib.save(isc_img,'isc.nii.gz')
# based on high isc res compute ISFC
isc_mask = (isc_r > 0.2)[:] 
print('Calculating ISFC on {0} voxels...'.format(np.sum(isc_mask)))
func_masked = all_masked_func[isc_mask, :, :]
isfc_r = isfc(func_masked)
print('Clustering ISFC...')
Z = linkage(isfc_r, 'ward')
z = fcluster(Z, 2, criterion='maxclust')
clust_inds = np.argsort(z)
# Show the ISFC matrix, sorted to show the two main clusters
plt.imshow(isfc_r[np.ix_(clust_inds, clust_inds)])
plt.show()