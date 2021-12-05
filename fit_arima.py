import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from getjson import getjson
from urllib.parse import urljoin
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from microprediction import MicroReader
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from warnings import catch_warnings, filterwarnings

from fitspec import FIT_PATH, FIT_URL, FitSpec

nlags = 400
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
    df.index = df.index.to_period("5T")
    return df

def select_stream():
    mr = MicroReader()
    prizes = mr.get_prizes()
    sponsors = random.choice([item['sponsor'] for item in prizes])
    sponsored = mr.get_sponsors()
    sponsored = [x for x in sponsored if '~' not in x]
    return random.choice(list(sponsored))

def make_grid():
    ps = [0, 1, 2, 3, 6]
    ds = [0, 1, 2]
    qs = [0, 1, 2]
    order_grid = []
    for p in ps:
        for d in ds:
            for q in qs:
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
    history = [x for x in train]

    for i in range(len(test)):
        yhat = arima_forecast(data, order)
        predictions.append(yhat)
        history.append(test[i])

    error = measure_rmse(test, predictions)
    return error

def score_model(data, order, debug=False):
    result = None
    key = order

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
    return (key, result)

def grid_search(data, grid, parallel=True):
    """
    Perform grid search over orders in grid against data.
    ntest significes 
    """
    scores = []
    print(f"Searching grid with {len(data)} elements.")
    if parallel:
        executor = Parallel(n_jobs=-2, verbose=10)
        tasks = (delayed(score_model)(data, order) for order in grid)
        results = executor(tasks)
        scores = results
    else:
        scores = [score_model(data, order) for order in grid]
    return scores

def get_work(stream=None):
    while True:
        if not stream:
            stream = select_stream()
        print(f"Selected stream {stream}")
        df = df_from_lagged(stream)
        if len(df) < 100:
            print(f"Not enough lag values {len(df)}, skipping")
            stream = None
            continue
        if len(np.unique(df.values)) < 0.3 * len(df.values):
            print(f"Quantized data, skipping")
            stream = None
            continue
        break
    spec_url = urljoin(FIT_URL, stream)
    print(f"Trying spec url: {spec_url}")
    spec = getjson(spec_url)
    if spec:
        print(f"Got spec from URL {spec}")
        spec = FitSpec(*spec)
    else:
        grid = make_grid()
        print("Could not find spec, initializing.")
        spec = FitSpec(stream, 400, grid, [], [])
    return df, spec

def main(args):
    print(args)
    workers = cpu_count() - 1
    nlags = 400


    df, spec = get_work(args.stream)
    spec._replace(numlags = nlags)
    ntodo = len(spec.todo)
    if ntodo <1:
        print(f"Nothing to do for this stream.")
        return 0
    ndone = len(spec.results)
    ntotal = ndone + ntodo
    print(f"Spec for {spec.stream}:\nProgress: {ndone}:{ntodo} {ndone/ntotal*100:.2f}%")
    if nlags > len(df):
        nlags = len(df)
    k = min(ntodo, workers)
    scores = grid_search(df['y'].values[:nlags], random.choices(spec.todo, k=k))
    scores = sorted(scores, key=lambda x: x[1])
    best_order = scores[0]
    print(f"{spec.stream} : best order = {best_order}")

    # merge results into spec, remove the processed orders from todo, and write it out.
    for k in scores:
        try:
            spec.todo.remove(k[0])
        except ValueError:
            pass
        spec.results.append(k)
    spec.tstamp.append(time.time())
    fname = os.path.join(FIT_PATH, spec.stream)
    print(spec)
    with open(fname, 'w+') as fp:
        json.dump(spec, fp)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument("-n", "--nlags", action="store", dest="nlags", help="Number of laggged values to use for fitting.")
    parser.add_argument("-s", "--stream", action="store", dest="stream", help="Stream to evaluate.")
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
