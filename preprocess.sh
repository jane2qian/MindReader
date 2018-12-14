
#%%
#!/bin/bash
#transport data from local machine
#local_path="~/Desktop/Dicom/"
#scp -i ~/.ssh/neuro-key.pem -rC $local_path ubuntu@10.0.1.48:/data/neuro/MindReader/sourcedata## instead of using fmriprep
export home_folder="/data/neuro/Pilot_sep"
cd $home_folder
#scaffold dset struc
sudo dcm2bids_scaffold -o $home_folder/FC_practice
#dicom in sourcedata path
Dicom_folder=${home_folder}/prisma_1000
declare -a subj_id=("01" "02") 
declare -a subjects=("sub-01" "sub-02" )
export SUBJECTS_DIR=$home_folder/FC_practice/sourcedata
#create config.json for dset easy from online json editor
for i in ${subj_id[@]};do
    sudo dcm2bids -d $Dicom_folder/${subjects[@]} -p ${i} -c $home_folder/pipeline/config.json -o $home_folder/FC_practice/sourcedata
done

#individual preprocessing steps
#define subjectname file in subject dir
#cd $SUBJECTS_DIR/derivates
#surface-based anatomical preprocessing
for i in ${subjects[@]};do
     3dWarp -deoblique $SUBJECTS_DIR/${i}/anat/sub*T1w.nii.gz -prefix warped-
done
#for i in ${subjects[@]};do
#     echo ${i}> $SUBJECTS_DIR/${i}/subjectname
#    #B0 correction
#    #epidewarp.fsl 
#    mkdir $subj 
#    recon-all -s $subj -i $SUBJECTS_DIR -all
    #FS to AFNI
#    @SUMA_Make_Spec_FS -sid $subj
#done
mkdir -p $home_folder/FC_practice/derivatives/anat
sudo docker run -ti --rm                                    \
         -v $SUBJECTS_DIR:/bids_dataset:ro                  \
	   -v $home_folder/FC_practice/derivatives/anat:/outputs                \
	   -v /data/neuro/license_freesurfer.txt:/license.txt \
         bids/freesurfer                                    \
         /bids_dataset /outputs participant --participant_label ${subj_id[@]} \
	   --license_file "/license.txt"                                        
&
@SUMA_Make_Spec_FS -sid ${subj_id[@]}

# func prep in afni
homedir=$SUBJECTS_DIR
indir="$homedir"
otdir="$home_folder/FC_practice/derivatives"
#MetroTS.py -r *.resp -p 50 -n 52 -v 1 -prefix sub*-regressor
for subj_id in ${subjects[@]};do
      subj_indir=$indir/$subj_id
      subj_otdir=$otdir/$subj_id
      if [-d $subj_otdir]
      then
           echo "$subj_otdir dir exist..."
      else
           mkdir -p $subj_otdir
      fi
      cd $subj_otdir
      cat > run.afni_proc << EOF
      afni_proc.py \
         -blocks tcat despike ricor tshift align tlrc volreg surf blur scale regress \
         -dsets $subj_indir/func/sub-0*_task-narrative-0*_bold.nii.gz \
         -copy_anat $subj_indir/anat/SUMA/brain.nii \
         -surf_anat $subj_indir/anat/SUMA/*_SurfVol.nii \
         -surf_spec $subj_indir/anat/SUMA/[1,2]_?h.spec \
         -tcat_remove_first_trs 10 \
         -ricor_regs $subj_indir/RICOR/r*.slibase.1D \
         -ricor_regress_method across-runs \
         -ricor_regs_nfirst 3 \
         -align_opts_aea -cost lpc+ZZ \
         -volreg_align_to MIN_OUTLIER \
         -volreg_align_e2a \
         -volreg_tlrc_warp \
         -volreg_warp_dxyz 3 \
         -blur_size 4 \
         -regress_motion_per_run \
         -regress_censor_motion 0.3 \
#         -regress_bandpass 0.0222 0.18 \
         -regress_opts_3dD \
         -jobs 4 \
         -bash -execute
      cd ../
     EOF
     sh run.afni_proc
done

