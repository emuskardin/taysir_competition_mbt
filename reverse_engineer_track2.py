import shutil
import zipfile

import mlflow
from aalpy.base import SUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWMethodEqOracle

from utils import get_validation_data

model_ids = [1, 2, 3, 4, 5, 8]


def load_function(path_to_zip):
    # unzip in tmp folder
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall('tmp')

    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
    unwrapped_model = loaded_model.unwrap_python_model()

    # execute few sequences to check if loading worked
    print('Model loaded, testing it on 10 sequences')
    # remove tmp subfolder
    shutil.rmtree('tmp/')
    return unwrapped_model


class ReverseEngineerSUL(SUL):
    def __init__(self, predict_fun):
        super().__init__()
        self.predict_fun = predict_fun

    def pre(self):
        self.string = []

    def post(self):
        pass

    def step(self, letter):
        if letter is not None:
            self.string.append(letter)
        return self.predict_fun.predict(None, ([start_symbol] + self.string + [end_symbol]))

    def query(self, word: tuple) -> list:
        if word is None:
            return self.predict_fun.predict(None, (start_symbol,) + (end_symbol,))
        return self.predict_fun.predict(None, (start_symbol,) + word + (end_symbol,))


for model_id in model_ids:
    print('---------------------------------------------')
    print('Model id')
    predict = load_function(f'tmp1/model_2_{model_id}.zip')
    model_name = f"models/2.{model_id}.taysir.model"

    _, start_symbol, end_symbol = get_validation_data(2, model_id)
    input_al = list(range(start_symbol))

    sul = ReverseEngineerSUL(predict)
    eq_oracle = RandomWMethodEqOracle(input_al, sul, walks_per_state=10, walk_len=20)

    model = run_KV(input_al, sul, eq_oracle, automaton_type='moore', )

    bin_size = [s.output for s in model.states]
    print(f'Num Bins: {len(set(bin_size))}')
