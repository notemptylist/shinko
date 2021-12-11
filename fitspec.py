import os
from collections import namedtuple
from typing import List, TypedDict

from numpy.lib.arraysetops import isin
FIT_URL = 'https://raw.githubusercontent.com/notemptylist/shinko/main/modelfits/arima/'
FIT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'modelfits',
                        'arima')
FitSpec = namedtuple('FitSpec', ['stream', 'numlags', 'todo', 'results', 'tstamp'])

class fitresult(TypedDict):
    order: tuple
    rmse: float
    mean: float
    tstamp: float

class fitspec(TypedDict):
    stream: str
    numlags: int
    todo: List[tuple]
    results: List[fitresult]

def make_spec():
    s: fitspec = {'stream': '',
                  'numlags': 0,
                  'todo': [],
                  'results': [],
                  }
    return s

if __name__ == "__main__":
    import json
    fs: fitspec = {'stream': 'foo.json',
                   'numlags': 400,
                   'todo': [(0, 0, 1), (1, 6, 0)],
                   'results': [
                       {'order': (1, 1, 1),
                        'rmse': .90,
                        'mean': .20,
                        'tstamp': 12312312312
                        }, ]
                   }
    print(fs)
    print(isinstance(fs, dict))
    with open('foo.json', 'w') as fp:
        json.dump(fs, fp)

    with open('foo.json', 'r') as fp:
        foo = json.load(fp)
        print(foo)
        print(isinstance(foo, dict))