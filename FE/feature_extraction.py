import os
import sys
from PIL import Image
import argparse
from pathlib import Path
import numpy as np
from bob.bio.vein.extractor import MaximumCurvature
from bob.bio.base.transformers import PreprocessorTransformer
from bob.bio.vein.preprocessor import NoCrop, NoMask, NoNormalization, NoFilter, Preprocessor

EXPECTED_FF_SIZE=600000 #614528

# For pre-processing. We didn't use any pre-processing in this work, just for change the input format for feeding the feature extraction
prep = Preprocessor(crop=NoCrop(), mask=NoMask(), normalize=NoNormalization(), filter=NoFilter())

# Maximum Curvature (MC) feature extraction function
fe = MaximumCurvature()

# For checking/making folder
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Feature extraction function.
def FE(name_imgs, path_fe, path_db):
    '''
    Args:
        name_imgs: name of input images
        path_fe: path for save the features
        path_db: path of the database
    '''
    num_imgs = len(name_imgs)
    for i, name_img in enumerate(name_imgs):
        name_f = 'mc_' + os.path.splitext(name_img)[0] # name of feature
        path_f = os.path.join(path_fe, name_f) # path of feature
        my_outf = path_f+'.npy' 
        # If the file already exist, then skip it. If not, extract the feature
        if os.path.isfile(my_outf) and os.path.getsize(my_outf) > EXPECTED_FF_SIZE:
            print('Nothing to do; feature-file already exists')
        else:
            path_img = os.path.join(path_db, name_img)
            img = Image.open(path_img)
            img = np.array(img)
            pre_img = prep(img)
            f = fe(pre_img) 
            np.save(path_f, f)
        print(i+1, '/', num_imgs)
    
    return


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--path_db', type=str, help='path of database', required=True)
    parser.add_argument('-fe','--path_fe', type=str, help='path of feature', required=True)
    args = parser.parse_args()

    path_db = args.path_db
    path_fe = args.path_fe
    
    #mkdir(path_fe)
    if not os.path.exists(path_fe):
        os.makedirs(path_fe)
    name_imgs = sorted(os.listdir(path_db))
    
    #exclude files from path_db that do not end in .bmp .
    fv_files = []
    for f in name_imgs:
        if f.endswith('.bmp'):
            fv_files.append(f)
    
    num_files = len(fv_files)
    print('Total number of fingervein files to process:', num_files)
    
    # MFFV-N has 57600 files, but grid does not allow more than 7500 tasks in one submission.
    # make chunks of 10 files; process each chunk as 1 job on grid
    chunk_size = 10
    chunks = [fv_files[x:x+chunk_size] for x in range(0, len(fv_files), chunk_size)]  
    num_chunks = len(chunks)
    print('Number of chunks:', num_chunks)
    
    # if idiap grid is available use it to distribute tasks
    task_id = 0
    num_tasks = 1
    if 'SGE_TASK_ID' in os.environ and os.environ['SGE_TASK_ID'] != 'undefined':
        task_id = int(os.environ['SGE_TASK_ID'])
        num_tasks = int(os.environ['SGE_TASK_LAST'])
        print('SGE_stats:', task_id, num_tasks)

        if task_id > num_chunks:
            assert 0, 'Grid request for job %d on a setup with %d jobs' % (task_id, num_chunks)
        file_set = chunks[task_id-1]
    else:
        # idiap grid not being used
        file_set = fv_files

    print('Num. fingervein files:', len(file_set))
    
    FE(name_imgs=file_set, path_fe=path_fe, path_db=path_db)



if __name__ == '__main__':
    main(sys.argv)
