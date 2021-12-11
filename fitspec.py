import os
from collections import namedtuple
from typing import Type, TypedDict
FIT_URL = 'https://raw.githubusercontent.com/notemptylist/shinko/main/modelfits/arima/'
FIT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'modelfits',
                        'arima')
FitSpec = namedtuple('FitSpec', ['stream', 'numlags', 'todo', 'results', 'tstamp'])
fitdict = {'stream': '',
           'numlags': 0,
           'todo' : [],
           'results': [],
           'tstamp': []
           }

class fitresult(TypedDict):
    order: tuple
    rmse: float
    mean: float
    tstamp: float

class fitspec(TypedDict):
    stream: str
    numlags: int
    todo: list[tuple]
    results: list[fitresult]


if __name__ == "__main__":
    import json
    fs = FitSpec('foo.json', 400, [(0,0,1), (1, 6, 0)], [ ((1,1,1), .90) ], [])
    print(fs)
    with open('foo.json', 'w') as fp:
        json.dump(fs._asdict(), fp)

    with open('foo.json', 'r') as fp:
        foo = json.load(fp)
        print(foo)
        fit = FitSpec(**foo)
        print(fit)
