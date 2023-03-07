from aalpy.base import Oracle
from aalpy.base.SUL import CacheSUL
from sklearn.metrics import mean_squared_error


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
    def __init__(self, alphabet: list, sul, oracle, validation_seq, start_symbol, end_symbol):
        super().__init__(alphabet, sul)
        self.oracle = oracle
        self.validation_sequances = validation_seq
        self.sul = sul
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

    def prune_cex(self, hypothesis, seq):
        is_cache_sul = isinstance(self.sul, CacheSUL)

        trimmed_cex = seq[1:-1]
        for i in range(1, len(trimmed_cex) + 1):
            tmp = trimmed_cex[:i]
            model_input = [self.start_symbol] + tmp + [self.end_symbol]

            if is_cache_sul:
                model_output = self.sul.sul.get_model_output(model_input)
            else:
                model_output = self.sul.get_model_output(model_input)

            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, tmp)[-1]
            if model_output != hyp_o:
                return tmp

        assert False

    def find_cex(self, hypothesis):
        is_cache_sul = isinstance(self.sul, CacheSUL)

        for seq in self.validation_sequances:
            if is_cache_sul:
                model_output = self.sul.sul.get_model_output(seq)
            else:
                model_output = self.sul.get_model_output(seq)
            hyp_o = hypothesis.execute_sequence(hypothesis.initial_state, seq[1:-1])[-1]

            if model_output != hyp_o:
                return self.prune_cex(hypothesis, seq)

        if self.oracle:
            return self.oracle.find_cex(hypothesis)
        else:
            return None


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
