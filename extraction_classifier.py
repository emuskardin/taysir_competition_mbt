import sys
from os import path

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import load_automaton_from_file

from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_classification_model

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


class BinaryRNNSUL(SUL):
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
                return model.predict(model.one_hot_encode([start_symbol,end_symbol]))
        else:
            self.current_word.append(letter)

        self.current_word.append(end_symbol)
        encoded_word = self.rnn.one_hot_encode(self.current_word)
        prediction = self.rnn.predict(encoded_word)

        self.current_word.pop(-1)

        return bool(prediction)

    def get_model_output(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return bool(self.rnn.predict(encoded_word))


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

    sul = BinaryRNNSUL(model)

    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=30)
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)

    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, None, validation_data, start_symbol, end_symbol)

    load = False
    learned_model_name = f'learned_models/track_{TRACK}_dataset_{DATASET}_model.dot'

    if load and path.exists(learned_model_name):
        learned_model = load_automaton_from_file(learned_model_name, 'dfa')
    else:
        learned_model = run_KV(input_alphabet, sul, validation_oracle, 'dfa', max_learning_rounds=10,)

    accuracy = test_accuracy_of_learned_classification_model(model, learned_model, validation_data)

    if accuracy > 0.99 and not load:
        learned_model.save(f'learned_models/track_{TRACK}_dataset_{DATASET}_model')


    def predict(seq):
        pruned_seq = [i for i in seq if i not in {start_symbol, end_symbol}]
        return learned_model.execute_sequence(learned_model.initial_state, pruned_seq)[-1]


    save_function(predict, nb_letters, f'dataset{TRACK}.{DATASET}', start_symbol, end_symbol,
                  submission_folder='submission_data')
