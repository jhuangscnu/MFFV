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
def get_all_scores(path=None, pt=None, st=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        pt: protocol, balance or normal
        st: set type, dev or test
    Returns:
        all_scores: authentication scores of all the three views
    '''
    path_csv = path
    all_scores = []
    for c in ['c1', 'c2', 'c3']:
        csv = st + '_' + pt + '_' + c + '.csv'
        scores = get_scores(path=path_csv, csv=csv)
        all_scores.append(scores)
    return all_scores

# function for getting the finger ID
def get_id(path=None, name=None, pt=None, st=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py 
        name: finger name, probe_subject_id or bio_ref_subject_id
        pt: protocol, balance or normal
        st: set type, dev or test
    Returns:
        data[name]: finger index
    '''
    csv = st + '_' + pt + '_c1' + '.csv'
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
def svm_fusion(path=None, kernel=None, pt=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        kernel: kernel of SVM
        pt: protocol
    Returns:
        scores_dev[:, 1]: SVM fusion scores of dev set
        scores_test[:, 1]: SVM fusion scores of test set
    '''
    all_scores_dev = get_all_scores(path=path, pt=pt, st='dev')
    all_scores_dev = numpy.array(all_scores_dev).transpose()

    all_scores_test = get_all_scores(path=path, pt=pt, st='test')
    all_scores_test = numpy.array(all_scores_test).transpose()

    csv_dev = 'dev_' + pt + '_c1' + '.csv'
    target_dev = get_svm_target(path=path, csv=csv_dev)

    fusion = svm.SVC(kernel=kernel, probability=True)
    fusion.fit(all_scores_dev, target_dev)

    scores_dev = fusion.predict_proba(all_scores_dev)

    scores_test = fusion.predict_proba(all_scores_test) # use the SVM trained on dev set for test set
    return scores_dev[:, 1], scores_test[:, 1]

# function for saving the SVM fusion scores
def svm_score_fusion(path=None, path_save=None, kernel=None, csv_title=None, pt=None):
    '''
    Args:
        path: path of the results (csv files) of experiments fvia.py
        path_save: path we want to save the fusion results
        kernel: kernel of SVM
        csv_title: title in the csv file
        pt: protocol
    '''
    probe_subject_id_dev = get_id(path=path, name='probe_subject_id', pt=pt, st='dev')
    bio_ref_subject_id_dev = get_id(path=path, name='bio_ref_subject_id', pt=pt, st='dev')

    probe_subject_id_test = get_id(path=path, name='probe_subject_id', pt=pt, st='test')
    bio_ref_subject_id_test = get_id(path=path, name='bio_ref_subject_id', pt=pt, st='test')

    scores_dev, scores_test = svm_fusion(path=path, kernel=kernel, pt=pt)

    data_save_dev = zip(*[numpy.array(probe_subject_id_dev), numpy.array(bio_ref_subject_id_dev), scores_dev])
    csv_save_dev = 'dev_' + pt + '_' + 'svm' + '_' + kernel + '.csv'
    save_csv(csv_data=data_save_dev, csv_path=path_save, csv_name=csv_save_dev, csv_title=csv_title)

    data_save_test = zip(*[numpy.array(probe_subject_id_test), numpy.array(bio_ref_subject_id_test), scores_test])
    csv_save_test = 'test_' + pt + '_' + 'svm' + '_' + kernel + '.csv'
    save_csv(csv_data=data_save_test, csv_path=path_save, csv_name=csv_save_test, csv_title=csv_title)

# function for scores fusion
def run_socre_level_fusion(path_single, path_fusion):
    '''
    Args: 
        path_single: path of the results (csv files) of experiments fvia.py
        path_fusion: path we want to save the fusion scores
    '''
    for pt in ['balance', 'nom']:
        svm_score_fusion(path=path_single, path_save=path_fusion, kernel='poly', csv_title=['probe_subject_id', 'bio_ref_subject_id', 'score'], pt=pt)
        print('protocol: ', pt, 'done!')
    print('score level fusion finished!')


def main(arguments):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True) # path of the results (csv files) of experiments fva_baseline.py
    parser.add_argument('-fs','--path_fs', type=str, help='path of fusion results', required=True) # path we want to save the fusion scores
    args = parser.parse_args()
    
    run_socre_level_fusion(path_single=args.path_rs, path_fusion=args.path_fs)

if __name__ == '__main__':
    main(sys.argv)





