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


def main(args):
    # load selected algorithms & hyperparameters from string or JSON file
    assert (args.string is None) != (args.file is None), 'Exactly one of --string and --file must be specified.'
    if args.string:
        configs = json.loads(args.string)
    elif args.file:
        with open(args.file) as f:
            configs = json.load(f)
    assert set(configs.keys()) == {'algorithms', 'hyperparameters'}, 'Invalid arguments.'

    # convert lists of hyperparameters to numpy arrays
    for alg in configs['algorithms']:
        for key, val in configs['hyperparameters'][alg].items():
            configs['hyperparameters'][alg][key] = np.array(val)

    # load training dataset
    dataset = pd.read_csv(args.data, header=None).values
    # TODO: try/except if dataset does not contain number
    dataset_id = int(re.findall("\\d+", args.data.split('/')[-1].split('.')[0])[0])
    t0 = time.time()
    x = dataset[:, :-1]
    y = dataset[:, -1]

    settings = util.generate_settings(configs['algorithms'], configs['hyperparameters'])
    headings = [str(s) for s in settings]
    results = np.zeros((2, len(settings)))

    # generate error matrix entries, i.e. compute k-fold cross validation error
    for i, setting in enumerate(settings):
        model = Model(args.p_type, setting['algorithm'], setting['hyperparameters'], args.verbose)
        start = time.time()
        cv_errors, _ = model.kfold_fit_validate(x, y, n_folds=args.n_folds)
        results[:, i] = np.array([cv_errors.mean(), time.time() - start])
        save_path = os.path.join(args.save_dir, str(dataset_id).zfill(5) + '.csv')
        pd.DataFrame(results, columns=headings, index=['Error', 'Time']).to_csv(save_path)

    # log results
    elapsed = time.time() - t0
    line = '\nID={}, Size={}, Time={:.0f}s, Avg. Error={:.3f}'\
           .format(dataset_id, dataset.shape, elapsed, results[0, :].mean())
    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as log:
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
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds to use for k-fold cross validation.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to generate print statements on completion.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
