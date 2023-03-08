from aalpy.base import Oracle
from aalpy.base.SUL import CacheSUL
from aalpy.utils.HelperFunctions import all_prefixes
from aalpy.utils import get_Angluin_dfa
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


class ValidationDataOracleWrapper(Oracle):
    def __init__(self, alphabet: list, sul, oracle, validation_seq, start_symbol, end_symbol, test_prefixes=False):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.sul = sul
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.test_prefixes = test_prefixes

        if test_prefixes:
            self.test_seq = set()
            for val_seq in validation_seq:
                self.test_seq.update(all_prefixes(val_seq[1:-1]))
            self.test_seq = list(self.test_seq)
            self.test_seq.sort(key=len)
        else:
            self.test_seq = [tuple(v[1:-1]) for v in validation_seq]

    def find_cex(self, hypothesis):
        to_remove = []
        test_seq_prime = self.test_seq.copy()

        for seq in tqdm(test_seq_prime):
            model_output = self.sul.get_model_output((self.start_symbol,) + seq + (self.end_symbol,))
            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, seq)[-1]

            if model_output != hyp_o:
                for t in to_remove:
                    self.test_seq.remove(t)

                if not self.test_prefixes:
                    return self.prune_cex(hypothesis, seq)
                else:
                    return seq

            if self.test_prefixes:
                to_remove.append(seq)

        if self.oracle:
            return self.oracle.find_cex(hypothesis)
        else:
            return None

    def prune_cex(self, hypothesis, seq):

        for i in range(1, len(seq) + 1):
            tmp = seq[:i]
            model_input = (self.start_symbol,) + tmp + (self.end_symbol,)

            model_output = self.sul.get_model_output(model_input)

            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, tmp)[-1]
            if model_output != hyp_o:
                return tmp

        assert False


def get_output(model, seq):
    current_state = model['q0'] # test
    for i in seq:
        current_state = model[current_state[1][i]]
    return current_state[0]


def test_accuracy_of_learned_classification_model(sul, automata, validation_sequances):
    num_tests = len(validation_sequances)
    num_positive_outputs = 0

    for seq in validation_sequances:
        model_prediction = sul.get_model_output(seq)
        output = get_output(automata, seq)
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

if __name__ == '__main__':
    s = get_Angluin_dfa()
    s = s.to_state_setup()

    o = get_output(s, ['a', 'b'])
    print(o)