import sys
from collections import Counter
from statistics import mean

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar, run_GSM
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import convert_i_o_traces_for_RPNI

from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_classification_model, \
    load_validation_data_outputs, test_sul_stepping

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


class BinaryRNNSUL(SUL):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        self.current_word = []
        self.hs = None

    # used only in eq. oracle
    def pre(self):
        self.current_word = []
        # Stepping
        # self.current_word = [start_symbol]
        # _, self.hs = self.rnn.forward(self.rnn.one_hot_encode(self.current_word), None)

    def post(self):
        pass

    # used only in the learning oracle that is not a validation oracle
    def step(self, letter):
        if letter is not None:
            self.current_word.append(letter)
        return self.get_model_output([start_symbol] + self.current_word + [end_symbol])
        # # Stepping
        # if letter is not None:
        #     _, self.hs = self.rnn.forward(self.rnn.one_hot_encode([letter]), self.hs)
        #
        # output, _ = self.rnn.forward(self.rnn.one_hot_encode([end_symbol]), self.hs)
        #
        # return bool(output.argmax(-1).flatten() >= 0.5)

    # helper method to get an output of a model for a sequence
    def get_model_output(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return bool(self.rnn.predict(encoded_word))

    # def query(self, word: tuple) -> list:
    #     self.num_queries += 1
    #     return [self.get_model_output((start_symbol,) + word + (end_symbol,))]


if __name__ == '__main__':

    # disable all gradients
    torch.set_grad_enabled(False)
    # binary classification
    track = 1
    not_solved_ids = [8, 11]

    for dataset in not_solved_ids:
        model_name = f"models/{track}.{dataset}.taysir.model"

        model = mlflow.pytorch.load_model(model_name)
        model.eval()

        if dataset != 7:  # RNN
            nb_letters = model.input_size - 1
            cell_type = model.cell_type
            print("The alphabet contains", nb_letters, "symbols.")
            print("The type of the recurrent cells is", cell_type.__name__)
        else:  # Transformer
            nb_letters = model.distilbert.config.vocab_size - 2
            print("The alphabet contains", nb_letters, "symbols.")
            print("The model is a transformer (DistilBertForSequenceClassification)")

        validation_data, start_symbol, end_symbol = get_validation_data(track, dataset)
        val_data_mean_len = int(mean([len(x) - 2 for x in validation_data]))

        sul = BinaryRNNSUL(model)

        input_alphabet = list(range(nb_letters))
        input_alphabet.remove(start_symbol)
        input_alphabet.remove(end_symbol)

        validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

        appears_in_all = set(input_alphabet)
        freq_counter = Counter()
        for inputs, outputs in validation_data_with_outputs.items():
            if outputs[-1]:
                for i in inputs[1:-1]:
                    freq_counter[i] += 1

        for x in freq_counter.most_common():
            print(x)

        pos, neg = 0, 0
        for inputs, outputs in validation_data_with_outputs.items():
            if outputs[-1] and pos < 10:
                print(f'True: {inputs[1:-1]}')
                pos+=1
            if not outputs[-1] and neg < 10:
                print(f'False: {inputs[1:-1]}')
                neg+=1

        current_input = []
        while True:
            inp = int(input())
            if inp == -1:
                current_input = []
                continue
            current_input.append(inp)
            prediction = sul.get_model_output([start_symbol,] + current_input + [end_symbol])
            print(f'Output for {current_input} : {prediction}')

        exit()
        #input_alphabet = [i[0] for i in freq_counter.most_common(n=3)]
        #
        eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100,
                                       min_walk_len=5,
                                       max_walk_len=5,
                                       reset_after_cex=True)

        learned_model = run_Lstar(input_alphabet, sul, eq_oracle=eq_oracle,
                                  automaton_type='dfa',
                                  max_learning_rounds=4,
                                  cache_and_non_det_check=False)

        learned_model.visualize()
