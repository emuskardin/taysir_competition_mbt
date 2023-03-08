import os
import pickle

from aalpy.base import Oracle
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

        with open(output_dataset_name, 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading outputs for prefixes of all validation sequances.')
        with open(output_dataset_name, 'rb') as handle:
            output_dict = pickle.load(handle)

    return output_dict


class ValidationDataOracleWrapper(Oracle):
    def __init__(self, alphabet: list, sul, oracle, validation_seq, track, dataset, start_symbol, end_symbol):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.sul = sul
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

        self.validation_seq_output_map = load_validation_data_outputs(sul, validation_seq, track, dataset)

    def find_cex(self, hypothesis):

        validation_input_sequances = list(self.validation_seq_output_map.keys())
        for seq in validation_input_sequances:
            hypothesis.reset_to_initial()
            outputs = self.validation_seq_output_map[seq]

            inputs = []
            for index, i in enumerate(seq[1:-1]):
                inputs.append(i)
                model_output = outputs[index]
                hyp_o = hypothesis.step(i)

                if model_output != hyp_o:
                    return tuple(inputs)

        if self.oracle:
            return self.oracle.find_cex(hypothesis)
        else:
            return None


def get_output(model, seq):
    current_state = model['s0']  # test
    for i in seq:
        current_state = model[current_state[1][i]]
    return current_state[0]


def test_accuracy_of_learned_classification_model(sul, automata, validation_sequances):
    num_tests = len(validation_sequances)
    num_positive_outputs = 0

    for seq in validation_sequances:
        model_prediction = sul.get_model_output(seq)
        seq = seq[1:-1]  # Remove start and end symbol
        output = automata.execute_sequence(automata.initial_state, seq)[-1]
        if output == bool(model_prediction):
            num_positive_outputs += 1

    accuracy = num_positive_outputs / num_tests
    print(f'Learned model accuracy on validation set: {round(accuracy, 4)}')
    return accuracy


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
