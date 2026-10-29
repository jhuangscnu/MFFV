import numpy
import pandas as pd
import os
import sys
from sklearn import svm
import argparse

def score_fusion_strategy(strategy_name="average"):
    """Returns a function to compute a fusion strategy between different scores.

    Different strategies are employed:

    * ``'average'`` : The averaged score is computed using the :py:func:`numpy.average` function.
    * ``'min'`` : The minimum score is computed using the :py:func:`min` function.
    * ``'max'`` : The maximum score is computed using the :py:func:`max` function.
    * ``'median'`` : The median score is computed using the :py:func:`numpy.median` function.
    * ``None`` is also accepted, in which case ``None`` is returned.
    """
    try:
        return {
            "average": numpy.average,
            "min": min,
            "max": max,
            "median": numpy.median,
            None: None,
        }[strategy_name]
    except KeyError:
        warn("score fusion strategy '%s' is unknown" % strategy_name)
        return None

# function for getting the authentication socres from the results we have saved
def get_scores(path=None, csv=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        csv: name of the input csv file
    Returns:
        data['score']: authentication scores
    '''
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    return data['score']

# function for getting the authentication socres of all the three views
def get_all_scores(path=None, mode=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        mode: mode of the illumination selection, see the m setting in fvia.py
    Returns:
        all_scores: authentication scores of all the three views
    '''
    path_csv = path
    all_scores = []
    for c in ['c1', 'c2', 'c3']:
        csv = 'dev_balance_' + c + '_' + mode + '.csv'
        scores = get_scores(path=path_csv, csv=csv)
        all_scores.append(scores)
    return all_scores

# function for getting the finger ID
def get_id(path=None, name=None, mode=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        name: finger name, probe_subject_id or bio_ref_subject_id
        mode: mode of the illumination selection, see the m setting in fvia.py
    Returns:
        data[name]: finger index
    '''
    csv = 'dev_balance_c1_' + mode + '.csv'
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    return data[name]

# function for save the data to a csv file
def save_csv(csv_data, csv_path, csv_name, csv_title):
    '''
    Args:
        csv_data: data we want to save in csv file
        csv_path: path we want to save the csv file
        csv_name: name of the csv file we want to save
        csv_title: title in the csv file
    '''
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    # path_csv = csv_path + csv_name
    path_csv = os.path.join(csv_path, csv_name)
    csv = pd.DataFrame(columns=csv_title, data=csv_data)
    csv.to_csv(path_csv, index=False)

# function for direct-scores-fusion
def direct_scores_fusion(path=None, strategy=None, path_save=None, csv_title=None, mode=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        strategy: scores-fusion strategy, average, min, max, median
        path_save: path we want to save the fusion socres
        csv_title: title in the csv file
        mode: mode of the illumination selection, see the m setting in fvia.py
    Returns:
        finnal_scores: fusion scores
    '''
    probe_subject_id = get_id(path=path, name='probe_subject_id', mode=mode)
    bio_ref_subject_id = get_id(path=path, name='bio_ref_subject_id', mode=mode)
    finnal_scores = []

    all_scores = get_all_scores(path=path, mode=mode)
    all_scores = numpy.array(all_scores)
    fusion = score_fusion_strategy(strategy_name=strategy)

    for num in range(all_scores.shape[1]):
        scores = all_scores[:, num]
        finnal_score = fusion(scores)
        finnal_scores.append(finnal_score)

    csv_save =  'dev_balance_' + mode +'_' + strategy + '.csv'
    data_save = zip(*[numpy.array(probe_subject_id), numpy.array(bio_ref_subject_id), finnal_scores])
    save_csv(csv_data=data_save, csv_path=path_save, csv_name=csv_save, csv_title=csv_title)
    return finnal_scores

# function for getting the label for training the SVM
def get_svm_target(path=None, csv=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        csv: name of the csv file
    Returns:
        target: target for training SVM
    '''
    path_csv = os.path.join(sys.path[0], path, csv)
    data = pd.read_csv(path_csv)
    target = []
    for idx in range(len(data)):
        if data['probe_subject_id'][idx] != data['bio_ref_subject_id'][idx]:
            target.append(0)
        else:
            target.append(1)
    return target

# function for SVM scores-fusion
def svm_fusion(path=None, kernel=None, mode=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        kernel: kernel of SVM
        mode: mode of the illumination selection, see the m setting in fvia.py
    Returns:
        scores_dev[:, 1]: SVM fusion scores
    '''
    all_scores = get_all_scores(path=path, mode=mode)
    all_scores = numpy.array(all_scores).transpose()
    csv = 'dev_balance_c1_' + mode + '.csv'
    target = get_svm_target(path=path, csv=csv)

    fusion = svm.SVC(kernel=kernel, probability=True)
    fusion.fit(all_scores, target)
    scores_dev = fusion.predict_proba(all_scores)
    return scores_dev[:, 1]

# function for saving the SVM fusion scores
def svm_score_fusion(path=None, path_save=None, kernel=None, csv_title=None, mode=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        path_save: path we want to save the fusion results
        kernel: kernel of SVM
        csv_title: title in the csv file
        mode: mode of the illumination selection, see the m setting in fvia.py
    '''
    probe_subject_id = get_id(path=path, name='probe_subject_id', mode=mode)
    bio_ref_subject_id = get_id(path=path, name='bio_ref_subject_id', mode=mode)

    scores = svm_fusion(path=path, kernel=kernel, mode=mode)
    data_save = zip(*[numpy.array(probe_subject_id), numpy.array(bio_ref_subject_id), scores])

    csv_save = 'dev_balance_' + mode + '_' + 'svm' + '_' + kernel + '.csv'
    save_csv(csv_data=data_save, csv_path=path_save, csv_name=csv_save, csv_title=csv_title)

# function for scores fusion
def run_score_level_fusion(path_single, path_fusion):
    '''
    Args: 
        path_single: path of the results (csv files) of experiments fvia.py
        path_fusion: path we want to save the fusion scores
    '''
    # fixed light experiemnts
    for mode in ['fl1', 'fl2', 'fl3', 'fl4', 'fl5', 'fl6']: # ablation experiments
        print('running fixed light score-level fusion...')
        for strategy in ['average', 'min', 'max', 'median']:
            direct_scores_fusion(path=path_single, path_save=path_fusion, strategy=strategy, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'strategy ', strategy, 'done!')
        for kernel in ['linear', 'poly', 'rbf']:
            svm_score_fusion(path=path_single, path_save=path_fusion, kernel=kernel, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'svm with kernel ', kernel, 'done!')
    print('fixed light score-level fusion finished!')
    
    # indivisual measure experiments
    for mode in ['m1', 'm2', 'm3']:
        print('running indivisual measure score-level fusion...')
        for strategy in ['average', 'min', 'max', 'median']:
            direct_scores_fusion(path=path_single, path_save=path_fusion, strategy=strategy, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'strategy ', strategy, 'done!')
        for kernel in ['linear', 'poly', 'rbf']:
            svm_score_fusion(path=path_single, path_save=path_fusion, kernel=kernel, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'svm with kernel ', kernel, 'done!')
    print('indivisual measure score-level fusion finished!')
            
    # threshold selection experiments
    for mode in ['m6_th60', 'm6_th70', 'm6_th80', 'm6_th90', 'm6_th100', 'm6_th110', 'm6_th120', 'm6_th130', 'm6_th140', 'm6_th150', 'm6_th160', 'm6_th170', 'm6_th180', 'm6_th190', 'm6_th200']: 
        print('running threshold selection score-level fusion...')
        for strategy in ['average', 'min', 'max', 'median']:
            direct_scores_fusion(path=path_single, path_save=path_fusion, strategy=strategy, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'strategy ', strategy, 'done!')
        for kernel in ['linear', 'poly', 'rbf']:
            svm_score_fusion(path=path_single, path_save=path_fusion, kernel=kernel, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'svm with kernel ', kernel, 'done!')
    print('threshold selection score-level fusion finished!')
            
    # ablation experiments
    for mode in ['m7', 'm8', 'm9']: 
        print('running ablation score-level fusion...')
        for strategy in ['average', 'min', 'max', 'median']:
            direct_scores_fusion(path=path_single, path_save=path_fusion, strategy=strategy, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'strategy ', strategy, 'done!')
        for kernel in ['linear', 'poly', 'rbf']:
            svm_score_fusion(path=path_single, path_save=path_fusion, kernel=kernel, csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], mode=mode)
            print('mode ', mode, 'svm with kernel ', kernel, 'done!')
    print('ablation score-level fusion finished!')
    print('all the socre level fusion finished!')


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True) # path of the results (csv files) of experiments fvia.py
    parser.add_argument('-fs','--path_fs', type=str, help='path of fusion', required=True) # path we want to save the fusion scores
    args = parser.parse_args()
    
    run_score_level_fusion(path_single=args.path_rs, path_fusion=args.path_fs)

if __name__ == '__main__':
    main(sys.argv)






