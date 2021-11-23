import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from microprediction import MicroReader
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from warnings import catch_warnings, filterwarnings

FIT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'modelfits',
                        'arima')
__version__ ='0.0.1'
MR = MicroReader()

def df_from_lagged(name='die.json'):
    """ Turn lagged times and values into a pandas DataFrame
    """
    lagged = MR.get_lagged_values_and_times(name)
    times, values = reversed(lagged[1]), reversed(lagged[0])

    df = pd.DataFrame({'Date' : pd.Series(times), 'y' : pd.Series(values)})
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df = df.set_index(['Date'])
    return df

def select_stream():
    mr = MicroReader()
    prizes = mr.get_prizes()
    sponsors = random.choice([item['sponsor'] for item in prizes])
    sponsored = mr.get_sponsors()
    sponsored = [x for x in sponsored if '~' not in x]
    return random.choice(list(sponsored))

def make_grid(p_values, d_values, q_values):
    order_grid = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order_grid.append((p, d, q))
    return order_grid

def arima_forecast(data, order):
    """Create an ARIMA model, fit it and make a single out of sample prediction."""
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    yhat = model_fit.predict(len(data), len(data))
    return yhat[0]

def train_test_split(data):
    s = int(len(data) * 0.66)
    return data[0:s], data[s:]

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def walk_forward_validation(data, order):
    predictions = list()
    train, test = train_test_split(data)
    print(f"Split data: {len(train)} in train, {len(test)} in test.")
    history = [x for x in train]

    for i in range(len(test)):
        yhat = arima_forecast(data, order)
        predictions.append(yhat)
        history.append(test[i])

    error = measure_rmse(test, predictions)
    return error

def score_model(data, order, debug=False):
    result = None
    key = str(order)

    if debug:
        result = walk_forward_validation(data, order)
    else:
        try:
            with catch_warnings():
                filterwarnings('ignore')
                print(f"< {order}")
                result = walk_forward_validation(data, order)
        except Exception as ex:
            print(ex) 

    if result:
        print(f"> ARIMA({key}) {result}", flush=True)
    return {key: result}

def grid_search(data, grid, parallel=True):
    """
    Perform grid search over orders in grid against data.
    ntest significes 
    """
    scores = dict()
    print(f"Searching grid with {len(data)} elements.")
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, order) for order in grid)
        results = executor(tasks)
        for x in results:
            scores.update(x)
    else:
        scores.update({score_model(data, order) for order in grid})
    return scores

def main(args):
    print(args)
    nlags = 400

    ps = [0, 1, 3, 4, 6]
    ds = list(range(0, 3))
    qs = list(range(0, 3))

    while True:
        stream = select_stream()
        print(f"Selected stream {stream}")
        df = df_from_lagged(stream)
        if len(df) < 100:
            print(f"Not enough lag values {len(df)}, skipping")
            continue
        if len(np.unique(df.values)) < 0.3 * len(df.values):
            print(f"Quantized data, skipping")
            continue
        break

    if nlags > len(df):
        nlags = len(df)

    grid = make_grid(ps, ds, qs)
    scores = grid_search(df['y'].values[:nlags], grid)
    print(scores)
    scores = sorted(scores, key=scores.get)
    best_order = scores[0]
    print(f"{stream} : best order = {best_order}")
    fname = os.path.join(FIT_PATH, stream)
    with open(fname, 'w+') as fp:
        json.dump(scores, fp)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument("-n", "--nlags", action="store", dest="nlags", help="Number of laggged values to use for fitting.")
    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    sys.exit(main(args))
