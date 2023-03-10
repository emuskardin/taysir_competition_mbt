import sys
from os import path
from statistics import mean

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar, run_RPNI, run_GSM
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

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
        # initial_state = torch.zeros(1, self.rnn.neurons_per_layer)
        # self.current_word = [start_symbol]
        # with torch.no_grad():
        #    _, self.hs = self.rnn(self.rnn.one_hot_encode(self.current_word), initial_state)
        self.current_word = []

    def post(self):
        pass

    # used only in the learning oracle that is not a validation oracle
    def step(self, letter):
        with torch.no_grad():
            if letter is not None:
                self.current_word.append(letter)
            return self.get_model_output([start_symbol] + self.current_word + [end_symbol])
            # if letter is None and self.current_word == [start_symbol]:
            #     return model.predict(model.one_hot_encode([start_symbol, end_symbol]))
            #
            # if letter is not None:
            #     _, self.hs = self.rnn(self.rnn.one_hot_encode([letter]), self.hs)
            #
            # output, _ = self.rnn(self.rnn.one_hot_encode([end_symbol]), self.hs)
            #
            # return bool(output.argmax(-1).flatten() >= 0.5)

    # helper method to get an output of a model for a sequence
    def get_model_output(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return bool(self.rnn.predict(encoded_word))

    def query(self, word: tuple) -> list:
        self.num_queries += 1
        return [self.get_model_output((start_symbol,) + word + (end_symbol,))]


class BinaryTransformerSUL(SUL):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.current_word = []

    def pre(self):
        self.current_word = [start_symbol]

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            if not self.current_word:
                return self.get_model_output([start_symbol, end_symbol])
        else:
            self.current_word.append(letter)

        self.current_word.append(end_symbol)

        with torch.no_grad():
            encoded_word = torch.IntTensor([[1] + [a + 2 for a in self.current_word]])
            prediction = self.transformer(encoded_word)
            prediction = prediction.logits.argmax().item()

        self.current_word.pop(-1)

        return bool(prediction)

    def get_model_output(self, seq):
        with torch.no_grad():
            encoded_word = torch.IntTensor([[1] + [a + 2 for a in seq]])
            prediction = self.transformer(encoded_word)
            return bool(prediction.logits.argmax().item())

    # 1 : 0.0750300000 (possibly context free, 100 states 0.85 acc, 7700 states 0.92 acc)
    # 2 : 0
    # 3 : 0
    # 4 : 0
    # 5 : 0
    # 6 : 0.0000100000 (strong oracle finds many cexes, but len seems to be limited to 22?)
    # 7 : 0
    # 8 : 0.3267700042
    # 9 : 0.0074657818
    # 10: 0.0149315637
    # 11: 0.2909509845


if __name__ == '__main__':

    track = 1

    model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    perfect = [2, 3, 4, 5, 7]
    almost_perfect = [1, 6, 9, 10]
    not_solved_ids = [8, 11]

    to_sample = [3, 4, 5, 7, 8, 11]
    current_test = [9]

    for dataset in to_sample:
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

        if dataset != 7:
            sul = BinaryRNNSUL(model)
        else:
            sul = BinaryTransformerSUL(model)

        input_alphabet = list(range(nb_letters))
        input_alphabet.remove(start_symbol)
        input_alphabet.remove(end_symbol)

        # three different oracles, all can achieve same results depending on the parametrization
        eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=500, min_walk_len=20, max_walk_len=40,
                                       reset_after_cex=False)
        strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=100, walk_len=40)

        # test_sul_stepping(sul, input_alphabet, start_symbol, end_symbol)
        # exit()

        validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

        validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, eq_oracle,
                                                        validation_data_with_outputs,
                                                        start_symbol, end_symbol, test_prefixes=True)

        learned_model = run_KV(input_alphabet, sul, validation_oracle,
                               automaton_type='dfa',
                               max_learning_rounds=500,
                               cache_and_non_det_check=False)

        print(f'Testing model: Track 1, Model {dataset}: Model size {learned_model.size}')

        # to enable saving of large models due to pickle recursion limit
        compact_model = learned_model.to_state_setup()

        # test accuracy
        val_data_mean_len = mean([len(x) - 2 for x in validation_data])
        acc_val, acc_rand = test_accuracy_of_learned_classification_model(sul, compact_model,
                                                                          validation_data_with_outputs,
                                                                          num_random_sequances=1000,
                                                                          random_seq_len=val_data_mean_len // 1.5)


        def predict(seq):
            current_state = 's0'
            for i in seq[1:-1]:
                current_state = compact_model[current_state][1][i]
            return compact_model[current_state][0]


        save_function(predict, nb_letters, f'dataset{track}.{dataset}', start_symbol, end_symbol,
                      submission_folder='submission_data')
