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

from qboost import qboost_lambda_sweep
from datasets import make_blob_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run QBoost example")


    n_samples = 2000
    n_features = 10

    X, y = make_blob_data(n_samples=n_samples, n_features=n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    
    normalized_lambdas = np.linspace(0.0, 0.7, 11)
    lambdas = normalized_lambdas / n_features
    qboost, lam = qboost_lambda_sweep(X_train, y_train, lambdas)

    print('best lambda:', lam)
    print('selected features:', qboost.get_selected_features())

    print('score on test set:', qboost.score(X_test, y_test))
