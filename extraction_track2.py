import sys
from random import choice
from statistics import mean

import numpy as np
import torch
import mlflow
from os import path
from aalpy.base import SUL, Oracle
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import load_automaton_from_file
from aalpy.utils.HelperFunctions import all_prefixes
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from extraction import get_validation_data
from submit_tools import save_function

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


def cluster_over_validation_set(model, validation_set, num_clusters=256):
    start_symbol, end_symbol = validation_set[0][0], validation_set[0][-1]

    prefix_set = set()
    for seq in validation_set:
        prefixes = all_prefixes(seq[1:-1])
        prefix_set.update(prefixes)

    observed_outputs = list()
    for prefix in prefix_set:
        padded_seq = [start_symbol] + list(prefix) + [end_symbol]
        outcome = model.predict(model.one_hot_encode(padded_seq))
        observed_outputs.append(outcome)

    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(validation_set)

    print('Clustering function computed.')
    return k_means


def create_bins(model, validation_set, num_bins=100):
    start_symbol, end_symbol = validation_set[0][0], validation_set[0][-1]

    prefix_set = set()
    for seq in validation_set:
        prefixes = all_prefixes(seq[1:-1])
        prefix_set.update(prefixes)

    observed_outputs = list()
    for prefix in prefix_set:
        padded_seq = [start_symbol] + list(prefix) + [end_symbol]
        outcome = model.predict(model.one_hot_encode(padded_seq))
        observed_outputs.append(outcome)

    observed_outputs.sort()
    bin_size = len(observed_outputs) // num_bins

    bins = [observed_outputs[i:i + bin_size] for i in range(0, len(observed_outputs), bin_size)]

    bin_means = [mean(bin) for bin in bins]

    #for i in range(len(bins) - 1):
    #    assert bin_ranges[i][0] <= bin_ranges[i][1] <= bin_ranges[i + 1][0]
    print('Bins computed.')
    return bin_means, bins


def get_bin_id(obs, bins):
    diffs = []
    for bin_mean in bins:
        diffs.append(abs(obs - bin_mean))

    return diffs.index(min(diffs))


def predict_from_bin(bin_id, bins):
    return bins[bin_id]
    if bin_id == -1:
        bin_id = 0
    if bin_id == -2:
        bin_id = len(bins) - 1

    return choice(bins[bin_id])


class ValidationDataOracleWrapper(Oracle):
    def __init__(self, alphabet: list, sul: SUL, oracle, validation_seq):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.validation_sequances = validation_seq

    def prune_cex(self, hypothesis, seq):
        trimmed_cex = seq[1:-1]
        for i in range(1, len(trimmed_cex) + 1):
            tmp = trimmed_cex[:i]
            model_input = [start_symbol] + tmp + [end_symbol]
            model_output = sul.predict(model_input)
            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, tmp)[-1]
            if model_output != hyp_o:
                return tmp

        assert False

    def find_cex(self, hypothesis):
        for seq in self.validation_sequances:
            model_output = sul.predict(seq)
            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, seq[1:-1])[-1]

            if model_output != hyp_o:
                return self.prune_cex(hypothesis, seq)

        if self.oracle:
            return self.oracle.find_cex(hypothesis)
        else:
            return None


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
                initial_output = model.predict(model.one_hot_encode([start_symbol, end_symbol]))
                return get_bin_id(initial_output, self.bins)
        else:
            self.current_word.append(letter)

        self.current_word.append(end_symbol)
        encoded_word = self.rnn.one_hot_encode(self.current_word)
        prediction = self.rnn.predict(encoded_word)

        self.current_word.pop(-1)

        return get_bin_id(prediction, self.bins)

    def predict(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return self.rnn.predict(encoded_word)


def test_accuracy_of_learned_model(rnn, automata, bins, validation_sequances):
    correct, model_predictions = [], []

    for seq in validation_sequances:
        rnn_output = rnn.predict(rnn.one_hot_encode(seq))
        seq = seq[1:-1]  # Remove start and end symbol
        output = automata.execute_sequence(automata.initial_state, seq)[-1]
        model_output = predict_from_bin(output, bins)

        correct.append(rnn_output)
        model_predictions.append(model_output)

    mean_square_error = mean_squared_error(correct, model_predictions)
    print(f'MSE on validation set: {mean_square_error}')
    return mean_square_error


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
    bin_means, bins = create_bins(model, validation_data, num_bins=10)

    sul = RegressionRNNSUL(model, bin_means)

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=30)
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)
    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, None, validation_data)

    # 50 - 1.09 e-10
    # 10 - 6.2 e-11
    # 5 - 1.01 e-10
    learned_model = run_KV(input_alphabet, sul, eq_oracle, 'moore', max_learning_rounds=10)

    accuracy = test_accuracy_of_learned_model(model, learned_model, bin_means, validation_data)

    exit()

    def predict(seq):
        pruned_seq = [i for i in seq if i not in {start_symbol, end_symbol}]
        reached_bin = learned_model.execute_sequence(learned_model.initial_state, pruned_seq)[-1]
        return predict_from_bin(reached_bin, bin_means)


    save_function(predict, nb_letters, f'dataset{TRACK}.{DATASET}', start_symbol, end_symbol)
