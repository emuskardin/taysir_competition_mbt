import sys
import torch
import mlflow
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


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

# for i in range(nb_letters):
#
#     e = model.one_hot_encode([])
#     p = model.predict(e)
#     print(e, p)

sul = RnnSUL(model)

input_alphabet = list(range(nb_letters))
input_alphabet.remove(start_symbol)
input_alphabet.remove(end_symbol)

eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=30)
strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)

learned_model = run_KV(input_alphabet, sul, strong_eq_oracle, 'dfa', max_learning_rounds=30)
# learned_model.visualize()

accuracy = test_accuracy_of_learned_model(model, learned_model, validation_data)

if accuracy >= 0.99:
    state_setup = learned_model.to_state_setup()
