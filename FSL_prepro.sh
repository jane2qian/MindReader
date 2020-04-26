#%%
#!/bin/bash
#STEP1:preparation
#transport data from local machine
#local_path="~/Desktop/LLS/"
#scp -i ~/.ssh/neuro-key.pem -rC $local_path ubuntu@10.0.1.48:/data/neuro/LLS_audio
export home_folder="/data/neuro/LLS_audio"
export SUBJECTS_DIR=$home_folder/sourcedata
cd $home_folder

#scaffold dset struct
#sudo dcm2bids_scaffold -o $home_folder
#dicom in sourcedata path
Dicom_folder=${home_folder}/Dicom/LLS/
#create config.json for dset easy from online json editor
for i in 44 45 48;do
    sudo dcm2bids -d $Dicom_folder/sub0${i} -p 0${i} -c $home_folder/config.json -o $SUBJECTS_DIR
done


# fieldmapping
for i in 44 45 48;do
    cd $SUBJECTS_DIR/sub-0${i}
    sudo chmod ugo+wx * 
    cd fmap
    fslmerge -t All_b0 sub-0${i}_fmap_run-01_fmap.nii.gz sub-0${i}_fmap_run-02_fmap.nii.gz
    topup --imain=All_b0.nii.gz --datain=$home_folder/sourcedata/datain.txt --config=b02b0.cnf --out=fmap_spline  --fout=fmap --iout=unwarped
    fslmaths fmap.nii.gz -mul 6.28 fmap_rads
    fslmaths unwarped.nii.gz -Tmean B0_mag
    bet B0_mag B0_mag_brain
    fsl_prepare_fieldmap SIEMENS fmap_rads.nii.gz B0_mag_brain.nii.gz undistored_fmap 2.46 --nocheck
done

cd ../../
#fsl structure preprocessing
for i in 44 45 48;do
    sudo chmod ugo+wx $SUBJECTS_DIR/sub-0${i}/*
    cd $SUBJECTS_DIR/sub-0${i}/anat/ 
    fsl_anat -i sub-0${i}_T1w.nii.gz -o 0${i} --nosubcortseg
    bet 0${i}.anat/T1_to_MNI_lin.nii.gz 0${i}.anat/T1_to_MNI_lin_brain.nii.gz
    bet 0${i}.anat/T1_to_MNI_nonlin.nii.gz 0${i}.anat/T1_to_MNI_nonlin_brain.nii.gz 
done
 
#motion outlier
for i in 44 45 48;do
    for run in 01 02 03 04;do
        cd $SUBJECTS_DIR/sub-0${i}/func/
        fsl_motion_outliers -i sub-0${i}_task-narrative_run-${run}_bold_reorient.nii.gz -o run${run}_motion_outlier -s DVARS -p DVARS  
    done
done 

#BLOCK0:physiological evs

#generate fsf templates individually andprocessing FEAT_*.fsf

#change local machine path in pnm files
dir=/data/neuro/LLS_audio/sourcedata/PNM/
for i in 44 45 48;do
    for run in 01 02 03 04;do
        sed -e 's@/Users/ruiqing/Desktop/physio_pnm/@'$dir'@g' sub0${i}-run${run}_evlist.txt > sub-0${i}-run-${run}evlist.txt
    done
done
sub_043_run-04_evlist.txt
#combine 5 runs epi individually
for subj in ${subj_id[@]};do ls sub${subj}.run??.feat/filtered_func_data.nii.gz | paste -sd " " > funcs ;fslmerge -t sub${subj}.func.all.nii.gz `cat funcs`;done

# flirt resize MNI152 template to functional data
flirt -in MNI152 -ref functional -out mask 

# percent signal change
deriv_dir="/data/neuro/LLS_audio/derivatives/analysis01/"
compreh_dir="/data/neuro/LLS_audio/derivatives/analysis02/ISFC/comprehension/"
cd $deriv_dir
ls -d $deriv_dir/sub???.run03.feat | paste -sd " " > featdir3

featquery 28 `cat featdir3`  1 func.nii.gz SFG -a 10 -p -t 0.1 -s $compreh_dir/HarvardOxford_SFG_mask.nii.gz

#get clusters in Talairach space 
for run in 01 02 03 04;do autoaq -i isc_share_L${run}.nii.gz -a "Talairach Daemon Labels" -t 0.1 -u -p -o srm_L_Talairach.txt;done

