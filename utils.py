import os
import pickle
from random import choices, randint

from aalpy.base import Oracle
from aalpy.utils.HelperFunctions import all_prefixes
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def get_validation_data(track, dataset):
    file = f"datasets/{track}.{dataset}.taysir.valid.words"

    sequences = []
    ss, es = None, None
    with open(file) as f:
        f.readline()  # Skip first line (number of sequences, alphabet size)
        for line in f:
            line = line.strip()
            seq = line.split(' ')
            seq = [int(i) for i in seq[1:]]  # Remove first value (length of sequence) and cast to int
            if ss is None:
                ss, es = seq[0], seq[-1]
            else:
                assert ss == seq[0] and es == seq[-1]
            sequences.append(seq)

    return sequences, ss, es


def load_validation_data_outputs(sul, validation_data, track, dataset):
    output_dataset_name = f'datasets/{track}.{dataset}.taysir.validation.output_map.pickle'

    if not os.path.exists(output_dataset_name):
        print('Computing outputs for prefixes of all validation sequances.')

        output_dict = dict()
        for seq in tqdm(validation_data):
            sul.pre()
            outputs = []
            for i in seq[1:-1]:
                model_output = sul.step(i)
                outputs.append(model_output)

            output_dict[tuple(seq)] = outputs
            sul.post()

        with open(output_dataset_name, 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading outputs for prefixes of all validation sequances.')
        with open(output_dataset_name, 'rb') as handle:
            output_dict = pickle.load(handle)

    return output_dict


class ValidationDataOracleWrapper(Oracle):
    def __init__(self, alphabet: list, sul, oracle, validation_data_with_outputs, start_symbol, end_symbol,
                 prefix_closed=True):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.sul = sul
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.prefix_closed = prefix_closed

        self.validation_seq_output_map = validation_data_with_outputs
        self.validation_processed = False

    def find_cex(self, hypothesis):

        if not self.validation_processed:
            for input_seq, output_seq in tqdm(self.validation_seq_output_map.items()):

                hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, input_seq[1:-1])[-1]

                if self.prefix_closed or output_seq[-1] != hyp_o:
                    hypothesis.reset_to_initial()
                    inputs = []
                    for index, i in enumerate(input_seq[1:-1]):
                        inputs.append(i)
                        hyp_o = hypothesis.step(i)

                        if hyp_o != output_seq[index]:
                            return tuple(inputs)

            self.validation_processed = True
            print('All counterexamples found with prefix-closed validation set.')
            print('Switching to testing oracle.')

        if self.oracle:
            oracle_cex = self.oracle.find_cex(hypothesis)
            if oracle_cex is None:
                self.num_queries = self.oracle.num_queries
                self.num_steps = self.oracle.num_steps
            return oracle_cex
        else:
            return None


def get_compact_model_output(model, seq):
    current_state = 's0'
    for i in seq:
        current_state = model[current_state][1][i]
    return model[current_state][0]


def test_accuracy_of_learned_classification_model(sul, automata, validation_sequances,
                                                  num_random_sequances=0, random_seq_len=40):
    validation_seq = list(validation_sequances.keys())
    num_validation_tests = len(validation_seq)
    ss, es = validation_seq[0][0], validation_seq[0][-1]
    input_alphabet = list(range(ss))

    num_positive_outputs_validation, num_positive_outputs_random = 0, 0

    for input_seq, model_outputs in validation_sequances.items():
        output = get_compact_model_output(automata, input_seq[1:-1])
        # output = automata.execute_sequence(automata.initial_state, input_seq[1:-1])[-1]
        if output == model_outputs[-1]:
            num_positive_outputs_validation += 1

    for _ in range(num_random_sequances):
        random_string = [ss] + choices(input_alphabet, k=randint(random_seq_len, random_seq_len * 2)) + [es]
        model_output = sul.get_model_output(random_string)
        output = get_compact_model_output(automata, random_string[1:-1])
        if output == model_output:
            num_positive_outputs_random += 1

    accuracy_val = num_positive_outputs_validation / num_validation_tests
    print(f'Learned model accuracy on validation test set: {round(accuracy_val, 4)}')

    if num_random_sequances > 0:
        accuracy_random = num_positive_outputs_random / num_random_sequances
        print(f'Learned model accuracy on random test set:     {round(accuracy_random, 4)}')

    return accuracy_val, accuracy_random


def test_accuracy_of_learned_regression_model(rnn, automata, bins, bin_predict_fun, validation_sequances):
    correct, model_predictions = [], []

    for seq in validation_sequances:
        rnn_output = rnn.predict(rnn.one_hot_encode(seq))
        seq = seq[1:-1]  # Remove start and end symbol
        output = automata.execute_sequence(automata.initial_state, seq)[-1]
        model_output = bin_predict_fun(output, bins)

        correct.append(rnn_output)
        model_predictions.append(model_output)

    mean_square_error = mean_squared_error(correct, model_predictions)
    print(f'MSE on validation set: {mean_square_error}')
    return mean_square_error


def test_sul_stepping(sul, input_al, ss, es):
    num_tests, num_disagreements = 0, 0
    for _ in tqdm(range(100)):
        seq = tuple(choices(input_al, k=20))
        prefixes = all_prefixes(seq)
        assert seq in prefixes

        for prefix in prefixes:

            num_tests += 1
            model_output = sul.get_model_output((ss,) + prefix + (es,))
            sul_output = sul.query(prefix)[-1]

            if model_output != sul_output:
                print(prefix)
                num_disagreements += 1

    print(f'Disagree on {num_disagreements} if {num_tests} tests.')


