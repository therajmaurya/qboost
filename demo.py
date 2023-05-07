#    Copyright 2018 D-Wave Systems Inc.

#    Licensed under the Apache License, Version 2.0 (the "License")
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http: // www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
try:
    import matplotlib.pyplot as plt
except ImportError:
    # Not required for demo
    pass

from qboost import QBoostClassifier, qboost_lambda_sweep
from datasets import make_blob_data, get_handwritten_digits_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run QBoost example",
                                     epilog="Information about additional options that are specific to the data set can be obtained using either 'demo.py blobs -h' or 'demo.py digits -h'.")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cross-validation', action='store_true',
                        help='use cross-validation to estimate the value of the regularization parameter')
    parser.add_argument('--lam', default=0.01, type=float,
                        help='regularization parameter (default: %(default)s)')

    # Note: required=True could be useful here, but not available
    # until Python 3.7
    subparsers = parser.add_subparsers(
        title='dataset', description='dataset to use', dest='dataset')

    sp_blobs = subparsers.add_parser('vanilla', help='run vanilla implementation on the new data')

    args = parser.parse_args()

    if args.dataset == 'vanilla':

        X_train, y_train, X_test, y_test = make_blob_data()

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        n_informative = X_train.shape[1] # keeping informative columns same as original number of columns

        qboost = QBoostClassifier(X_train, y_train, args.lam)

        print('Informative features:', list(range(n_informative)))
        print('Selected features:', qboost.get_selected_features())

        print('Score on test set: {:.3f}'.format(qboost.score(X_test, y_test)))

    elif not args.dataset:
        parser.print_help()
