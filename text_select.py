
# coding: utf-8

import os
import csv
import wave
import contextlib

# based on bits choose audio(size <= 10MB)
chosen_audio_dir = ('/Users/ruiqing/Desktop/Material_11_21')
source_stm_dir = ('/Users/ruiqing/Downloads/TEDLIUM_release2/train/stm/')

def main():
    dirName = chosen_audio_dir;
    stm_name=[]
    WordCount=[]
    duration=[]
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    audiopath = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += filenames
        audiopath += [os.path.join(dirpath, file) for file in filenames] 
        print audiopath
        for audio in audiopath:
            if audio.endswith("wav"):
                with contextlib.closing(wave.open(audio,'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration.append((frames/float(rate))/60)

    for elem in listOfFiles:
        try:
            name_length=len(elem) 
            if elem.endswith("wav"): 
                stm_name.append(elem[0:(len(elem)-4)])
                stm_file=open(source_stm_dir + elem[0:(len(elem)-4)] + ".stm").read()
                WordCount.append(len(stm_file.split()))
                
                    
        except:
            pass 
    
    with open(os.path.join(chosen_audio_dir,'text_chosen.csv'), 'wb') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["stm_name", "Nword", "duration"])
        writer.writerows(zip(stm_name,WordCount,duration))

        
if __name__ == '__main__':
    main()

