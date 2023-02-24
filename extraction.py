import sys
import torch
import mlflow
from os import path
from aalpy.base import SUL, Oracle
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import load_automaton_from_file

from submit_tools import save_function

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


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


class RnnSUL(SUL):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.current_word = []

    def pre(self):
        self.current_word = [start_symbol]

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            if not self.current_word:
                # TODO empty sequence
                return False
        else:
            self.current_word.append(letter)

        self.current_word.append(end_symbol)
        encoded_word = self.rnn.one_hot_encode(self.current_word)
        prediction = self.rnn.predict(encoded_word)

        self.current_word.pop(-1)

        return bool(prediction)

    def predict(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return bool(self.rnn.predict(encoded_word))


def test_accuracy_of_learned_model(rnn, automata, validation_sequances):
    num_tests = len(validation_sequances)
    num_positive_outputs = 0

    for seq in validation_sequances:
        rnn_prediction = rnn.predict(rnn.one_hot_encode(seq))
        seq = seq[1:-1]  # Remove start and end symbol
        output = automata.execute_sequence(automata.initial_state, seq)[-1]
        if output == bool(rnn_prediction):
            num_positive_outputs += 1

    accuracy = num_positive_outputs / num_tests
    print(f'Learned model accuracy on validation set: {round(accuracy, 4)}')
    return accuracy


if __name__ == '__main__':

    TRACK = 1
    DATASET = 0

    model_name = f"models/{TRACK}.{DATASET}.taysir.model"

    model = mlflow.pytorch.load_model(model_name)
    model.eval()

    nb_letters = model.input_size - 1
    cell_type = model.cell_type

    validation_data, start_symbol, end_symbol = get_validation_data(TRACK, DATASET)

    print("The alphabet contains", nb_letters, "symbols.")
    print("The type of the recurrent cells is", cell_type.__name__)

    sul = RnnSUL(model)

    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=30)
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)

    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, None, validation_data)

    load = True
    learned_model_name = f'learned_models/track_{TRACK}_dataset_{DATASET}_model.dot'

    if not path.exists(learned_model_name) and load:
        learned_model = run_KV(input_alphabet, sul, validation_oracle, 'dfa', max_learning_rounds=100)
    else:
        learned_model = load_automaton_from_file(learned_model_name, 'dfa')

    accuracy = test_accuracy_of_learned_model(model, learned_model, validation_data)

    if accuracy > 0.99 and not load:
        learned_model.save(f'learned_models/track_{TRACK}_dataset_{DATASET}_model'[:-4])


    def predict(seq):
        return learned_model.execute_sequence(learned_model.initial_state, seq[1:-1])[-1]

    save_function(predict, nb_letters, f'dataset{TRACK}.{DATASET}', start_symbol, end_symbol)
