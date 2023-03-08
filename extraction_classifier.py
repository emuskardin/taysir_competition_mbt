import sys
from os import path

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_classification_model, \
    load_validation_data_outputs

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
        self.current_word = [start_symbol]
        with torch.no_grad():
            _, self.hs = self.rnn(self.rnn.one_hot_encode(self.current_word), None)

    def post(self):
        pass

    # used only in the learning oracle that is not a validation oracle
    def step(self, letter):
        with torch.no_grad():
            if letter is None:
                if not self.current_word:
                    return model.predict(model.one_hot_encode([start_symbol, end_symbol]))
            else:
                output, self.hs = self.rnn(self.rnn.one_hot_encode([letter]), self.hs)

            output, _ = self.rnn(self.rnn.one_hot_encode([end_symbol]), self.hs)

            return bool(output.argmax(-1).flatten() >= 0.5)

    # helper method to get an output of a model for a sequence
    def get_model_output(self, seq):
        encoded_word = self.rnn.one_hot_encode(seq)
        return bool(self.rnn.predict(encoded_word))


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


if __name__ == '__main__':

    track = 1
    model_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    perfect = [3, 4, 5, 7]
    almost_perfect = [2, 6, 9]
    not_solved_ids = [1, 8, 10, 11]

    current_test = [10]

    for dataset in current_test:
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
        eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=1000, min_walk_len=10, max_walk_len=50,
                                       reset_after_cex=False)
        strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=100, walk_len=40)

        validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

        validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, strong_eq_oracle,
                                                        validation_data_with_outputs,
                                                        start_symbol, end_symbol)

        learned_model = run_KV(input_alphabet, sul, validation_oracle,
                               automaton_type='dfa',
                               max_learning_rounds=1000,
                               cache_and_non_det_check=False)

        print(f'Testing model: Track 1, Model {dataset}: Model size {learned_model.size}')

        # to enable saving of large models due to pickle recursion limit
        compact_model = learned_model.to_state_setup()

        # test accuracy
        acc_val, acc_rand = test_accuracy_of_learned_classification_model(sul, compact_model,
                                                                          validation_data_with_outputs,
                                                                          num_random_sequances=1000,
                                                                          random_seq_len=len(validation_data[0]) - 2)

        def predict(seq):
            current_state = 's0'
            for i in seq[1:-1]:
                current_state = compact_model[current_state][1][i]
            return compact_model[current_state][0]


        save_function(predict, nb_letters, f'dataset{track}.{dataset}', start_symbol, end_symbol,
                      submission_folder='submission_data')
