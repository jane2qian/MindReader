
#%%
#!/bin/bash
#transport data from local machine
#local_path="~/Desktop/Dicom/"
#scp -i ~/.ssh/neuro-key.pem -rC $local_path ubuntu@10.0.1.48:/data/neuro/MindReader/sourcedata## instead of using fmriprep

export SUBJECTS_DIR="/data/neuro/MindReader"
#scaffold dset struc
sudo dcm2bids_scaffold -o $home_folder
#dicom in sourcedata path
Dicom_folder=${home_folder}/sourcedata

#individual preprocessing steps
declare -a SUBJECT=("sub-01" "sub-02" "sub-03")
#define subjectname file in subject dir
for i in ${SUBJECT[@]};do
    echo ${i}> $SUBJECTS_DIR/${i}/subjectname
done
cd $SUBJECTS_DIR
#B0 correction
#epidewarp.fsl 
#surface-based anatomical preprocessing
for subj in ${SUBJECT}; do
    recon-all -s $subj -all
    #FS to AFNI
    @SUMA_Make_Spec_FS -sid $subj
done

MetroTS.py -r *.resp -c card_file.dat -p 50 -n 52 -v 1 -prefix sub*-regressor
homedir="/data/neuro/Pilot_sep"
indir="$homedir/pipeline"
otdir="$homedir/derivates"

cd $indir
declare -a subjects=("sub-01" "sub-02")

for subj_id in ${subjects[@]};do
      subj_indir=$indir/$subj_id
      subj_otdir=$otdir/$subj_id/run-01
      mkdir -p $subj_otdir
      cd $subj_otdir
      cat > run.afni_proc << EOF
      afni_proc.py \
         -blocks tcat despike ricor tshift align tlrc volreg surf blur scale regress \
         -dsets $subj_indir/func/run-01/sub-02_task-listening_run-01_bold.nii \
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

