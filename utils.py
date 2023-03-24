import os
import pickle
from random import choices, randint

import numpy
import torch
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
            sequences.append(tuple(seq))

    sequences = list(set(sequences))

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
                 test_prefixes=True):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.sul = sul
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.test_prefixes = test_prefixes

        self.validation_seq_output_map = validation_data_with_outputs
        self.validation_processed = False

    def find_cex(self, hypothesis):

        if not self.validation_processed:
            for input_seq, output_seq in tqdm(self.validation_seq_output_map.items()):

                self.num_queries += 1
                self.num_steps += len(input_seq) - 2

                hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, input_seq[1:-1])[-1]

                if self.test_prefixes or output_seq[-1] != hyp_o:
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


class ValidationDataRegressionOracle(Oracle):
    def __init__(self, alphabet: list, sul, validation_data_with_outputs, start_symbol, end_symbol,
                 test_prefixes=True):
        super().__init__(alphabet, sul)
        self.sul = sul
        self.mapper = self.sul.mapper
        assert self.mapper
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.test_prefixes = test_prefixes

        self.validation_seq_output_map = validation_data_with_outputs

        self.abstract_input_output_map = dict()
        for inputs, outputs in self.validation_seq_output_map.items():
            self.abstract_input_output_map[inputs] = [self.mapper.to_abstract(o) for o in outputs]
        self.validation_processed = False

    def find_cex(self, hypothesis):

        if not self.validation_processed:
            for input_seq, output_seq in tqdm(self.abstract_input_output_map.items()):

                self.num_queries += 1
                self.num_steps += len(input_seq) - 2

                hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, input_seq[1:-1])[-1]

                if self.test_prefixes or output_seq[-1] != hyp_o:
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

        return None


def get_compact_model_output(model, seq):
    current_state = 's0'
    for i in seq:
        current_state = model[current_state][1][i]
    return model[current_state][0]


def test_accuracy_of_learned_classification_model(sul, automata, validation_data_with_outputs,
                                                  num_random_sequances=0, random_seq_len=40):
    validation_seq = list(validation_data_with_outputs.keys())
    num_validation_tests = len(validation_seq)
    ss, es = validation_seq[0][0], validation_seq[0][-1]
    input_alphabet = list(range(ss))

    num_positive_outputs_validation, num_positive_outputs_random = 0, 0

    for input_seq, model_outputs in validation_data_with_outputs.items():
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

    accuracy_random = -1
    if num_random_sequances > 0:
        accuracy_random = num_positive_outputs_random / num_random_sequances
        print(f'Learned model accuracy on random test set:     {round(accuracy_random, 4)}')

    return accuracy_val, accuracy_random


def test_accuracy_of_learned_regression_model(sul, automata, validation_data_with_outputs,
                                              num_random_sequances=0, random_seq_len=40):
    validation_seq = list(validation_data_with_outputs.keys())
    ss, es = validation_seq[0][0], validation_seq[0][-1]
    input_alphabet = list(range(ss))

    correct, model_predictions = [], []

    for input_seq, model_outputs in validation_data_with_outputs.items():
        if not model_outputs:
            continue

        output = get_compact_model_output(automata, input_seq[1:-1])

        correct.append(model_outputs[-1])
        model_predictions.append(output)

    correct_random, model_predictions_random = [], []

    for _ in range(num_random_sequances):
        random_string = [ss] + choices(input_alphabet, k=randint(random_seq_len, random_seq_len * 2)) + [es]
        model_output = sul.get_model_output(random_string)
        output = get_compact_model_output(automata, random_string[1:-1])
        correct_random.append(model_output)
        model_predictions_random.append(output)

    mse_validation = mean_squared_error(correct, model_predictions) * pow(10, 6)
    mse_random = -1
    print(f'MSE on validation set: {mse_validation}')
    if num_random_sequances > 0:
        mse_random = mean_squared_error(correct_random, model_predictions_random) * pow(10, 6)
        print(f'MSE on random set:     {mse_random}')
    return mse_validation, mse_random


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


def visualize_density(validation_data_with_outputs, consider_prefixes=True):
    import matplotlib.pyplot  as plt
    observed_outputs = set()

    for _, outputs in validation_data_with_outputs.items():
        if consider_prefixes:
            observed_outputs.update(outputs)
        else:
            observed_outputs.add(outputs[-1])

    observed_outputs = list(observed_outputs)
    observed_outputs.sort()

    x_plot = [i for i in range(len(observed_outputs))]

    plt.hist(observed_outputs, bins=len(observed_outputs) // 10)

    plt.show()


# Helper methods to execute a sequence and obtain output on transformers for regression

def make_future_masks(words: torch.Tensor):
    masks = (words != 0)
    b, l = masks.size()
    x = torch.einsum("bi,bj->bij", masks, masks)
    x *= torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
    x += torch.eye(l, dtype=torch.bool, device=x.device)
    return x.type(torch.int8)


def predict_next_symbols(model, word):
    """
    Args:
        whole word (list): a complete sequence as a list of integers
    Returns:
        the predicted probabilities of the next ids for all prefixes (2-D ndarray)
    """
    word = [[a + 1 for a in word]]
    word = torch.IntTensor(word)
    model.eval()
    with torch.no_grad():
        attention_mask = make_future_masks(word)
        out = model.forward(word, attention_mask=attention_mask)
        out = torch.nn.functional.softmax(out.logits[0], dim=1)
        return out.detach().numpy()[:, 1:]  # the probabilities for padding id (0) are removed


def predict_transformer(model, word):
    probs = predict_next_symbols(model, word[:-1])
    probas_for_word = [probs[i, a] for i, a in enumerate(word[1:])]
    value = numpy.array(probas_for_word).prod()
    return float(value)
