# MindReader
 LLS-neuro-project00 
 
+ experiment materials are in audio_sample folder with a description file
+ two preprocessing scripts. one is using fmriprep from docker and the other is single process specified. 
    > fieldmap(not collect yet). respiration and heartrate data(ongoing).
+ text_selection uses stm files of the TED audio downloaded from [TEDLIUM_release2](http://www.openslr.org/19/) ongoing.

+ fmriprep preprocessing steps: 
     + anatomical: 
        > 1.Intensity non-uniformity correction @n4; skull-stripped OASIS template ANTs 2.2.0
        > 2.Surface reconstruction, brain mask estimation @mindboggle FreeSurfer 6.0.1
        > 3.Spatial normalization to ICBM 152 Nonlinear Asymmetrical template 2009c. ANTs 2.2.0
        > 4.Segmentation fsl-fast FSL 5.0.9
     + functional:
        > 1.@bbr registration BOLD to T1w 
        > 2.motion correction @mcflirt FSL 5.0.9
        > 3.slice timing
        > 4.Normalization to MNI152NLin2009cAsym(1*1*1) standard space, surface resampling to fsnative
   
