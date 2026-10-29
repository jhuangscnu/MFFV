import os
import sys
import argparse
from PIL import Image
from PIL import ImageDraw
import cv2 as cv
import numpy as np
import math
import shutil
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from bob.bio.vein.extractor import MaximumCurvature
from bob.bio.vein.extractor import RepeatedLineTracking
from bob.bio.vein.extractor import WideLineDetector
from bob.bio.base.transformers import PreprocessorTransformer
from bob.bio.vein.preprocessor import NoCrop, NoMask, NoNormalization, NoFilter, Preprocessor
from bob.bio.vein.algorithm.MiuraMatch import MiuraMatch

# For getting the ROI of FVIA
def get_region(path_img=None, camera=None, seg=None): 
    '''
    Args:
        path_img: path of input image
        camera: camera index
        seg: if use segmentation or not
    Returns:
        reg: segmented ROI 
    '''
    img = Image.open(path_img)#.covert('L')
    roi_width  = 240
    roi_height = 60
    if seg:
        if camera == 1: 
            #reg = img.crop((50, 80, 290, 140)) # left, upper, right, lower
            left=50
            upper=80
        elif camera == 2:
            #reg = img.crop((50, 70, 290, 130)) # left, upper, right, lower
            left=50
            upper=70
        else:
            #reg = img.crop((50, 80, 290, 140)) # left, upper, right, lower
            left=50
            upper=80
        reg = img.crop((left, upper, left+roi_width, upper+roi_height)) # left, upper, right, lower
    else:
        reg = img
    #print('cropped reg. size:', reg.size)
    return reg

# For calculating entropy
def cal_etp_old(img=None):
    '''
    Args:
        img: input image (ROI)
    Returns:
        etp: value  of entropy
    '''
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    etp = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k) # probability of each pixel value
    for i in range(len(tmp)):
        if tmp[i] ==0:
            etp = etp
        else:
            etp = float(etp - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return etp

# For calculating entropy
def cal_etp(img=None):
    '''
    Args:
        img: input image (ROI)
    Returns:
        etp: value  of entropy
    '''
    img = Image.fromarray(img)
    etp = img.entropy()
    return etp

# For calculating variance
def cal_var(img=None):
    '''
    Args:
        img: input image (ROI)
    Returns:
        var: value of variance
    '''
    var = np.var(img)
    return var

# For calculating mean gray
def cal_avg(img=None):
    '''
    Args:
        img: input image (ROI)
    Returns:
        avg: value of mean gray
    '''
    avg = np.average(img)
    return avg


# FVIA algorithm
def fvqa(etp=None, var=None, avg=None, th=None):
    '''
    Args:
        etp: value of entropy
        var: value of variance
        avg: value of mean gray
        th: value of threshold
    Returns:
        mt: value of FVIA metric
    '''
    if avg <= th:
        a = -0.5 * math.pi * (1-avg/th)**2 - math.pi/6
    else:
        a = 0.5 * math.pi * ((avg-th)/(256-th))**2 + math.pi/6
    mt = (math.exp(etp) + var) / math.log(abs(math.tanh(a)) + 1)
    return mt

# FVIA ablation for entropy
def fvqa_mode8(etp=None, var=None, avg=None, th=None):
    '''
    Args:
        etp: value of entropy
        var: value of variance
        avg: value of mean gray
        th: value of threshold
    Returns:
        mt: value of FVIA metric
    '''
    if avg <= th:
        a = -0.5 * math.pi * (1-avg/th)**2 - math.pi/6
    else:
        a = 0.5 * math.pi * ((avg-th)/(256-th))**2 + math.pi/6
    mt = (var) / math.log(abs(math.tanh(a)) + 1)
    return mt

# For getting the finger vein quality metric
def get_mt(path_img=None, camera=None, mode=None, th=None, seg=None):
    '''
    Args:
        path_img: path of input image
        camera: camera index
        mode: mode of metric
        th: threshold value in FVIA algorithm
        seg: whether to use ROI
    Returns:
        mt: finger vein quality metric
    '''
    reg = get_region(path_img=path_img, camera=camera, seg=seg)
    reg = np.array(reg)
    etp = cal_etp(img=reg)
    var = cal_var(img=reg)
    avg = cal_avg(img=reg)
    # finally metric
    if mode == 1: # entropy metric
        mt = etp
    elif mode == 2: # variance metric
        mt = var
    elif mode == 3: # mean gray metric
        mt = avg
    elif mode ==4: # mean of entropy, variance, and mean gray (we didn't use it in the this paper)
        mt = (etp + var + avg) / 3
    elif mode == 5: # product of entropy, variance, and mean gray (we didn't use it in the this paper)
        mt = etp * var * avg
    elif mode == 6:
        mt = fvqa(etp=etp, var=var, avg=avg, th=th) # FVIA metric
    elif mode == 7:
        mt = fvqa(etp=etp, var=0, avg=avg, th=th) # metric of FVIA ablation for variance
    elif mode == 8:
        mt = fvqa_mode8(etp=0, var=var, avg=avg, th=th) # metric of FVIA ablation for entropy
    elif mode == 9:
        mt = math.exp(etp) + var # metric of FVIA ablation for mean gray (we remove the whole dominator)
    else:
        print('mode does not exist!')
    return mt

# For getting the illumination index
def get_light(path_db=None, finger=None, angle=None, camera=None, sample=None, mode=None, th=None, postfix=None, seg=None):
    '''
    Args:
        path_db: path of database
        finger: finger identity
        angle: finger orientation (we didn't use it in this paper)
        camera: camera index
        sample: sample index (10 presentations, thus 10 samples)
        mode: finger vein quality metric mode
        th: threshold in FVIA algorithm
        postfix: postfix of the files
        seg: whether to use ROI segmentation
    Returns:
        best_light: index of the best illumination
    '''
    best_light = 0
    highest_mt = 0
    for light in range(1, 7):
        img = str(finger) + '_' + str(angle) + '_' + str(light) + '_' + str(camera) + '_' + str(sample) + postfix # image name
        #path_img = path_db + '/' + img
        path_img = os.path.join(path_db, img)
        mt = get_mt(path_img=path_img, camera=camera, mode=mode, th=th, seg=seg) # get the finger vein quality metric
        # save the best metric
        if mt > highest_mt:
            highest_mt = mt
            best_light = light
        else:
            pass
    if best_light == 0:
        print('light error')
    if highest_mt == 0:
        print('metric error')
    return best_light

# get reference image name
def get_reference_id(path_db=None, finger=None, sample=None, camera=None, angle=None, mode=None, th=None, postfix=None, fixed_light=None, seg=None):
    '''
    Args:
        path_db: path of database
        finger: finger identity
        sample: sample index
        camera: camera index
        angle: finger orientation index (we didn't use it in this work)
        mode: mode for getting the finger vein quality metric
        th: threshold value in FVIA algorithm
        postfix: postfix of file
        fixed_light: whether to use fixed illumination
        seg: whether to use ROI segmentation
    Returns:
        reference_id: reference image name
    '''
    if fixed_light == None: 
        light = get_light(path_db=path_db, finger=finger, angle=angle, camera=camera, sample=sample, mode=mode, th=th, postfix=postfix, seg=seg)
    else:
        light = fixed_light
    reference_id = str(finger) + '_' + str(angle) + '_' + str(light) + '_' + str(camera) + '_' + str(sample)
    return reference_id

# get the authentication pair from the protocol file (.csv file) according to the mode of finger vein quality assessment mode
class GetAuthenticationPair(Dataset):
    '''
    Args:
        path_db: path of database
        path_fe: path of features
        path_csv: path of protocol files 
        thansform: transform method (we didn't use it in this paper)
        csv_name: name of protocol file
        prefix: file prefix
        postfix: file postfix
        camera: camera index
        angle: finger orientation index
        mode: mode for finger vein quality assessment
        th: value of threshold in FVIA algorithm
        seg: whether to use ROI segmentation
        fixed_light: whether to use fixed illumination
    Return:
        img1: enrolled image
        img2: probe image
        label: label of genuine (1) or impostor (0). (we didn't use it in this work)
        probe_reference_id: name of probe image
        probe_subject_id: finger identity of probe image
        bio_ref_reference_id: name of enrolled image
        bio_ref_subject_id: finger identity of enrolled image
    ''' 
    def __init__(self, path_db=None, path_fe=None, path_csv = None, transform = None, csv_name = None, prefix=None, postfix = None, camera=None, angle=None, mode=None, th=None, seg=None, fixed_light=None):
        self.path_db = path_db
        self.path_fe = path_fe
        self.transform = transform
        self.csv_name = csv_name
        self.prefix = prefix
        self.postfix = postfix
        self.path_csv = os.path.join(sys.path[0], path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.camera = camera
        self.angle = angle
        self.mode = mode
        self.th = th
        self.seg = seg
        self.fixed_light = fixed_light

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        # get the name of enrolled image
        bio_ref_reference_id = get_reference_id(path_db=self.path_db, finger=self.csv['finger_enrolled'][index], sample=self.csv['sample_enrolled'][index], camera=self.camera, angle=self.angle, mode=self.mode, th=self.th, postfix=self.postfix, seg=self.seg, fixed_light=self.fixed_light)
        # get the finger identity of enrolled image
        bio_ref_subject_id = self.csv['finger_enrolled'][index]
        # get the name of probe image
        probe_reference_id = get_reference_id(path_db=self.path_db, finger=self.csv['finger_probe'][index], sample=self.csv['sample_probe'][index], camera=self.camera, angle=self.angle, mode=self.mode, th=self.th, postfix=self.postfix, seg=self.seg, fixed_light=self.fixed_light)
        # get the finger identity of probe image
        probe_subject_id = self.csv['finger_probe'][index]
        # construct the entire files name
        path_img1 = os.path.join(self.path_fe, self.prefix+'_' + bio_ref_reference_id + '.npy')  # add mc here
        path_img2 = os.path.join(self.path_fe, self.prefix+'_' + probe_reference_id   + '.npy') # add mc here
        # load the file, here we load the MC features
        img1 = np.load(path_img1)
        img2 = np.load(path_img2)
        # bale of genuine or impostor
        label = self.csv['label'][index]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label, probe_reference_id, probe_subject_id, bio_ref_reference_id, bio_ref_subject_id

# check/make the folder
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# For saving the scores file
def save_rs_csv(csv_data, csv_path, csv_name, csv_title):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    #path_csv = csv_path + csv_name
    path_csv = os.path.join(csv_path, csv_name)
    csv = pd.DataFrame(columns=csv_title, data=csv_data)
    csv.to_csv(path_csv, index=False)



# For conducting the MM matching
def run(path_db=None, path_fe=None, path_csv=None, path_save=None, save_csv=None, csv_name=None, fem='mc', postfix='.bmp', transform=None, camera=None, angle=1, mode=None, th=None, seg=True, fixed_light=None):
    '''
    Args:
        path_db: path of database
        path_fe: path of features
        path_csv: path of protocol files 
        path_save: path of results
        csv_name: name of protocol file
        fem: feature extraction method (here we fixed it as MC method)
        postfix: file postfix
        thansform: transform method (we didn't use it in this paper)
        camera: camera index
        angle: finger orientation index (we didn't use it in this paper)
        mode: mode for finger vein quality assessment
        th: value of threshold in FVIA algorithm
        seg: whether to use ROI segmentation
        fixed_light: whether to use fixed illumination
    ''' 
    print('running for: ', csv_name)
    # title of score file we want to save
    title = ['probe_reference_id', 'probe_subject_id', 'bio_ref_reference_id', 'bio_ref_subject_id', 'score']
    # for getting the dataset
    dataset = GetAuthenticationPair(path_db=path_db, path_fe=path_fe, path_csv=path_csv, transform = transform, csv_name = csv_name, prefix = fem, postfix = postfix, camera=camera, angle=angle, mode=mode, th=th, seg=seg, fixed_light=fixed_light)
    # build the data loader
    data_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)
    # matching method: miura match
    miura_match = MiuraMatch()
    # for saving the authentication scores
    result_data = []
    print('running baseline ', fem, 'for camera ', camera, 'with mode ', mode, 'and th ', th, 'with fixed light ', fixed_light)
    
    # load the authentication pair one by one and calculate the authentication score
    for img1, img2, label, probe_reference_id, probe_subject_id, bio_ref_reference_id, bio_ref_subject_id in tqdm(data_loader):
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = img1.numpy()
        img2 = img2.numpy()
        score = miura_match.score(model=img1, probe=img2)
        result_data.append([probe_reference_id[0], probe_subject_id.numpy()[0], bio_ref_reference_id[0], bio_ref_subject_id.numpy()[0], score])
    
    # save the authentication scores in a .csv file
    save_rs_csv(csv_data=result_data, csv_path=path_save, csv_name=save_csv, csv_title=title)
    print('expt. finished')
    return


#construct list of experiments that can be run in parallel on grid
def construct_experiment_list():
    experiment_list = []
    #experiments to select best threshold T based on fvqa()
    expt_tag = 'Threshold_Selection'
    m=6 #fvqa-mode
    l=None  #fixed_light is set to None
    for c in range(1,4): # for all three views
        for th in range(60,201,10): # for traversal threshold T
            #save_csv = 'ths'+'_dev_balance' +'_c'+ str(c) + '_m'+str(m) + '_th'+ str(th) + '.csv' # name of result file
            save_csv = 'dev_balance' +'_c'+ str(c) + '_m'+str(m) + '_th'+ str(th) + '.csv' # name of result file
            my_tuple = (c, m, th, l, save_csv, expt_tag)
            experiment_list.append(my_tuple)
    
    #experiments for fixed-illumination
    expt_tag = 'Fixed_Illumination'
    th=None
    m=None
    for l in range(1,7): # for all six fixed illuminations
        for c in range(1, 4): # for all three views
            #save_csv = 'fxl'+'_dev_balance' +'_c'+str(c) +'_fl'+str(l) +'.csv' # name of result file
            save_csv = 'dev_balance' +'_c'+str(c) +'_fl'+str(l) +'.csv' # name of result file
            my_tuple = (c, m, th, l, save_csv, expt_tag)
            experiment_list.append(my_tuple)
    
    # experiments for individual measures
    expt_tag = 'Individual_Measures'
    th=None
    l=None
    for m in range(1, 4): # for the individual measure mode
        for c in range(1, 4): # for all three views
            #save_csv = 'inm'+'_dev_balance' +'_c'+str(c) +'_m'+str(m) +'.csv' # name of result file
            save_csv = 'dev_balance' +'_c'+str(c) +'_m'+str(m) +'.csv' # name of result file
            my_tuple = (c, m, th, l, save_csv, expt_tag)
            experiment_list.append(my_tuple)
    
    # experiments for ablation
    expt_tag = 'Ablation'
    th=120
    l=None
    for m in range(7, 10): # for ablation mode
        for c in range(1, 4): # for all three views
            #save_csv = 'abl'+'_dev_balance' +'_c' + str(c) + '_m'+str(m) +'.csv' # name of result file
            save_csv = 'dev_balance' +'_c' + str(c) + '_m'+str(m) +'.csv' # name of result file
            my_tuple = (c, m, th, l, save_csv, expt_tag)
            experiment_list.append(my_tuple)
    
    return experiment_list

#----
def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--path_db', type=str, help='path of database', required=True)
    parser.add_argument('-fe','--path_fe', type=str, help='path of database', required=True)
    parser.add_argument('-pt','--path_pt', type=str, help='path of protocols', required=True)
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True)
    args = parser.parse_args()
    
    # Make a list of experiments to be run, and then run them in a single loop (or distributed on a grid).
    
    experiment_list = construct_experiment_list() #list containing parameter-tuples for various experiments
    num_expts = len(experiment_list)
    
     # if idiap grid is available use it to distribute tasks
    task_id = 0
    num_tasks = 1
    if 'SGE_TASK_ID' in os.environ and os.environ['SGE_TASK_ID'] != 'undefined':
        #grid is available -- distribute experiments on grid
        task_id = int(os.environ['SGE_TASK_ID'])
        num_tasks = int(os.environ['SGE_TASK_LAST'])
        print('SGE_stats:', task_id, 'of', num_tasks)
	
	
        if task_id > num_expts:
            assert 0, 'Grid request for job %d on a setup with %d jobs' % (task_id, num_expts)
        #expt_param_tuple = experiment_list[task_id-1]
        c, m, th, f_l, outfile, expt_tag = experiment_list[task_id-1] #expt_param_tuple
        print(expt_tag, '.#', task_id, 'of', num_expts,'for:', outfile)
        run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv',\
        path_save=args.path_rs, save_csv=outfile, camera=c, mode=m, th=th, fixed_light=f_l)
        
    else:
    	# no computing grid available -- run all experiments locally
    	
    	for task_id, expt_param_tuple in enumerate(experiment_list):
    	    #expt_param_tuple = experiment_list[task_id-1]
            c, m, th, f_l, outfile, expt_tag= expt_param_tuple
            #print('for: ', outfile)
            print(expt_tag, '.#', task_id+1, 'of', num_expts,'for:', outfile)
            run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv',\
            path_save=args.path_rs, save_csv=outfile, camera=c, mode=m, th=th, fixed_light=f_l)

#	# experiments for traversal threshold T
#	for c in range(1, 4): # for all three views
#	    for th in range(60, 201, 10): # for traversal threshold T
#	        save_csv = 'dev_balance_c' + str(c) + '_m6_th'+ str(th) + '.csv' # name of result file
#	        print('for: ', save_csv)
#	        run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv', path_save=args.path_rs, save_csv=save_csv, camera=c, mode=6, th=th, fixed_light=None)

#	# experiments for fixed illuminations
#	for l in range(1, 7): # for all six fixed illuminations
#	    for c in range(1, 4): # for all three views
#	        save_csv = 'dev_balance_c'+str(c)+'_fl'+str(l)+'.csv' # name of result file
#	        print('for: ', save_csv)
#	        run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv', path_save=args.path_rs, save_csv=save_csv, camera=c, mode=None, th=None, fixed_light=l)
	
#	# experiments for individual measures
#	for m in range(1, 4): # for the individual measure mode
#	    for c in range(1, 4): # for all three views
#	        save_csv = 'dev_balance_c' + str(c) + '_m'+ str(m) + '.csv' # name of result file
#	        print('for: ', save_csv)
#	        run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv', path_save=args.path_rs, save_csv=save_csv, camera=c, mode=m, th=None, fixed_light=None)
		    
#	# experiments for ablation
#	for m in range(7, 10): # for ablation mode
#	    for c in range(1, 4): # for all three views
#	        save_csv = 'dev_balance_c' + str(c) + '_m' + str(m) + '.csv' # name of result file
#	        print('for: ', save_csv)
#	        run(path_db=args.path_db, path_fe=args.path_fe, path_csv=args.path_pt, csv_name='dev_balance.csv', path_save=args.path_rs, save_csv=save_csv, camera=c, mode=m, th=120, fixed_light=None)
    

if __name__ == '__main__':
    main(sys.argv)
    


