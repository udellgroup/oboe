"""
Generate a row of the error matrix for a given dataset. Records cross-validation error & elapsed time for each
algorithm & hyperparameter combination.

Note the difference between model "configurations" and "settings": configurations is a nested dictionary, containing
a list of algorithms, and a dictionary of lists of hyperparameters; settings is a list of dictionaries, with one
algorithm and a dictionary of hyperparameters. Below is an example of each:

Config: {'algorithms': ['KNN', 'DT'],
         'hyperparameters': {'KNN': {'n_neighbors': [1, 3, 5, 7], 'p': [1, 2]},
                             'DT':  {'min_samples_split': [0.01, 0.001]}
                            }
        }

Settings: [{'algorithm': 'KNN', 'hyperparameters': {'n_neighbors': 1, 'p': 1}},
           {'algorithm': 'KNN', 'hyperparameters': {'n_neighbors': 3, 'p': 2}},
           {'algorithm': 'DT',  'hyperparameters': {'min_samples_split': 0.01}}
          ]
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import sys
import re
import time
import util
from model import Model
import mkl

mkl.set_num_threads(1)
RANDOM_STATE = 0

def main(args):
    # load selected algorithms & hyperparameters from string or JSON file
    assert (args.string is None) != (args.file is None), 'Exactly one of --string and --file must be specified.'
    if args.string:
        configs = json.loads(args.string)
    elif args.file:
        with open(args.file) as f:
            configs = json.load(f)
    assert set(configs.keys()) == {'algorithms', 'hyperparameters'}, 'Invalid arguments.'

    # load training dataset
    dataset = pd.read_csv(args.data, header=None).values
    filename = args.data.split('/')[-1].split('.')[0]
    # whether to use dataset filename as error matrix vector filename 
    if args.fullname:
        dataset_id = filename        
    else:
        dataset_id = int(re.findall("\\d+", filename)[0])        

    # do not generate error matrices twice on one dataset
    if args.error_matrix != None:
        if args.error_matrix.endswith('.csv'):
            generated_datasets = pd.read_csv(args.error_matrix, index_col=0).index.tolist()
            assert dataset_id not in generated_datasets, 'Already generated.'

    t0 = time.time()
    x = dataset[:, :-1]
    y = dataset[:, -1]

    settings = util.generate_settings(configs['algorithms'], configs['hyperparameters'])
    headings = [str(s) for s in settings]
    results = np.full((2, len(settings)), np.nan)

    # generate error matrix entries, i.e. compute k-fold cross validation error
    log_file = [file for file in os.listdir(args.save_dir) if file.startswith('log')][0]
    for i, setting in enumerate(settings):
        model = Model(args.p_type, setting['algorithm'], setting['hyperparameters'], args.auc, args.verbose)
        start = time.time()
        try:
            cv_errors, _ = model.kfold_fit_validate(x, y, n_folds=args.n_folds, random_state=RANDOM_STATE)
        except (ZeroDivisionError, KeyError, TypeError, ValueError) as e:
            with open(os.path.join(args.save_dir, log_file), 'a') as log:
                line = '\nID={}, model={}, {}'.format(dataset_id, setting, e)
                log.write(line)
        results[:, i] = np.array([cv_errors.mean(), time.time() - start])
        if args.fullname:
            save_path = os.path.join(args.save_dir, str(dataset_id) + '.csv')
        else:
            save_path = os.path.join(args.save_dir, str(dataset_id).zfill(5) + '.csv')
        pd.DataFrame(results, columns=headings, index=['Error', 'Time']).to_csv(save_path)

    # log results
    elapsed = time.time() - t0
    line = '\nID={}, Size={}, Time={:.0f}s, Avg. Error={:.3f}'\
           .format(dataset_id, dataset.shape, elapsed, results[0, :].mean())
    with open(os.path.join(args.save_dir, log_file), 'a') as log:
        log.write(line)
    print(line)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('p_type', type=str, help='Problem type. Either classification or regression.')
    parser.add_argument('data', type=str, help='File path to training dataset.')
    parser.add_argument('--string', type=str,
                        help='JSON-style string listing all algorithm types and hyperparameters. '
                             'See automl/util.py for example.')
    parser.add_argument('--file', type=str,
                        help='JSON file listing all algorithm types and hyperparameters. '
                             'See automl/defaults/models.json for example.')
    parser.add_argument('--save_dir', type=str, default='./custom',
                        help='Directory in which to save new error matrix.')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds to use for k-fold cross validation.')
    parser.add_argument('--verbose', type=lambda x: x == 'True', default=False,
                        help='Whether to generate print statements on completion.')
    parser.add_argument('--error_matrix', type=str, default=None,
                        help='Existing error matrix. Avoid re-generate its rows.')
    parser.add_argument('--auc', type=lambda x: x == 'True', default=False, help='Whether to use AUC instead of BER')
    parser.add_argument('--fullname', type=lambda x: x == 'True', default=False,
                        help='Whether to use the full name of dataset as corresponding error matrix vectors.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
