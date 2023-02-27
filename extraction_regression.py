import sys
from statistics import mean

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils.HelperFunctions import all_prefixes

from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_regression_model

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


def create_bins(rnn, validation_set, num_bins=100):
    start_symbol, end_symbol = validation_set[0][0], validation_set[0][-1]

    prefix_set = set()
    for seq in validation_set:
        prefixes = all_prefixes(seq[1:-1])
        prefix_set.update(prefixes)

    observed_outputs = list()
    for prefix in prefix_set:
        padded_seq = [start_symbol] + list(prefix) + [end_symbol]
        outcome = rnn.predict(rnn.one_hot_encode(padded_seq))
        observed_outputs.append(outcome)

    observed_outputs.sort()
    bin_size = len(observed_outputs) // num_bins

    bins = [observed_outputs[i:i + bin_size] for i in range(0, len(observed_outputs), bin_size)]
    bin_means = [mean(bin) for bin in bins]

    print('Bins computed.')
    return bin_means


def get_bin_id(obs, bins):
    diffs = []
    for bin_mean in bins:
        diffs.append(abs(obs - bin_mean))

    return diffs.index(min(diffs))


def predict_from_bin(bin_id, bins):
    return bins[bin_id]


class RegressionRNNSUL(SUL):
    def __init__(self, rnn, bins):
        super().__init__()
        self.rnn = rnn
        self.current_word = []
        self.bins = bins

    def pre(self):
        self.current_word = [start_symbol]

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            if not self.current_word:
                return model.predict(model.one_hot_encode([start_symbol, end_symbol]))
        else:
            self.current_word.append(letter)

        self.current_word.append(end_symbol)
        encoded_word = self.rnn.one_hot_encode(self.current_word)
        prediction = self.rnn.predict(encoded_word)

        self.current_word.pop(-1)

        return get_bin_id(prediction, self.bins)

    def get_model_output(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return self.rnn.predict(encoded_word)


if __name__ == '__main__':
    TRACK = 2
    DATASET = 0

    model_name = f"models/{TRACK}.{DATASET}.taysir.model"

    model = mlflow.pytorch.load_model(model_name)
    model.eval()

    nb_letters = model.input_size - 1
    cell_type = model.cell_type

    validation_data, start_symbol, end_symbol = get_validation_data(TRACK, DATASET)

    print("The alphabet contains", nb_letters, "symbols.")
    print("The type of the recurrent cells is", cell_type.__name__)

    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    load = False
    learned_model_name = f'learned_models/track_{TRACK}_dataset_{DATASET}_model.dot'

    # if load and path.exists(learned_model_name):
    #     learned_model = load_automaton_from_file(learned_model_name, 'mealy')
    # else:
    bin_means = create_bins(model, validation_data, num_bins=50)

    sul = RegressionRNNSUL(model, bin_means)

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=30)
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)
    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, None, validation_data, start_symbol,
                                                    end_symbol)

    learned_model = run_KV(input_alphabet, sul, eq_oracle, 'moore', max_learning_rounds=10)

    mse = test_accuracy_of_learned_regression_model(model, learned_model, bin_means,
                                                    predict_from_bin, validation_data)


    def predict(seq):
        pruned_seq = [i for i in seq if i not in {start_symbol, end_symbol}]
        reached_bin = learned_model.execute_sequence(learned_model.initial_state, pruned_seq)[-1]
        return predict_from_bin(reached_bin, bin_means)


    save_function(predict, nb_letters, f'dataset{TRACK}.{DATASET}', start_symbol, end_symbol,
                  submission_folder='submission_data')
