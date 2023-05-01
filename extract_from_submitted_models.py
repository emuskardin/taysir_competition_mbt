import shutil
import zipfile

import mlflow
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWMethodEqOracle

from utils import get_validation_data, ValidationDataOracleWrapper, load_validation_data_outputs

track = 1
model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def load_function(path_to_zip):
    # unzip in tmp folder
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall('tmp')

    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
    unwrapped_model = loaded_model.unwrap_python_model()

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
            return [self.predict_fun.predict(None, (start_symbol,) + (end_symbol,))]
        return [self.predict_fun.predict(None, (start_symbol,) + word + (end_symbol,))]


for model_id in [1]:
    if track == 2 and model_id == 11:
        continue

    print(f'submission_data/final_submission/model_{track}_{model_id}.zip')
    predict = load_function(f'submission_data/final_submission/model_{track}_{model_id}.zip')

    validation_data, start_symbol, end_symbol = get_validation_data(track, model_id)
    validation_data_with_outputs = load_validation_data_outputs(None, validation_data, track, model_id)

    input_al = list(range(start_symbol))

    sul = ReverseEngineerSUL(predict)
    # eq_oracle = ValidationDataOracleWrapper(input_al, sul, None,
    #                                                 validation_data_with_outputs,
    #                                                 start_symbol, end_symbol, test_prefixes=True)
    eq_oracle = RandomWMethodEqOracle(input_al, sul, walks_per_state=100, walk_len=75)
    model = run_Lstar(input_al, sul, eq_oracle,
                      automaton_type='dfa' if track == 1 else 'moore',
                      cache_and_non_det_check=False)

    if track == 2:
        num_partitions = len(set([s.output for s in model.states]))
        print(f'Model {model_id} num partitions: {num_partitions}')

    model.save(f'learned_models/model_{track}_{model_id}')
