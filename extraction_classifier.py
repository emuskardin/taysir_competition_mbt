import sys
from os import path

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar
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
                return model.predict(model.one_hot_encode([start_symbol, end_symbol]))
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

    model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    TRACK = 1

    model_ids = [11]  #

    for DATASET in model_ids:
        model_name = f"models/{TRACK}.{DATASET}.taysir.model"

        model = mlflow.pytorch.load_model(model_name)
        model.eval()

        if DATASET != 7:  # RNN
            nb_letters = model.input_size - 1
            cell_type = model.cell_type
            print("The alphabet contains", nb_letters, "symbols.")
            print("The type of the recurrent cells is", cell_type.__name__)
        else:  # Transformer
            nb_letters = model.distilbert.config.vocab_size - 2
            print("The alphabet contains", nb_letters, "symbols.")
            print("The model is a transformer (DistilBertForSequenceClassification)")

        validation_data, start_symbol, end_symbol = get_validation_data(TRACK, DATASET)

        if DATASET != 7:
            sul = BinaryRNNSUL(model)
        else:
            sul = BinaryTransformerSUL(model)

        input_alphabet = list(range(nb_letters))
        input_alphabet.remove(start_symbol)
        input_alphabet.remove(end_symbol)

        # three different oracles, all can achieve same results depending on the parametrization
        eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=40,
                                       reset_after_cex=False)
        strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=20)

        validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, eq_oracle, validation_data, start_symbol,
                                                        end_symbol)

        load = False
        learned_model_name = f'learned_models/track_{TRACK}_dataset_{DATASET}_model.dot'

        if load and path.exists(learned_model_name):
            learned_model = load_automaton_from_file(learned_model_name, 'dfa')
        else:
            learned_model = run_KV(input_alphabet, sul, validation_oracle, 'dfa', max_learning_rounds=100)

        print(f'Testing model: Track 1, Model {DATASET}: Model size {learned_model.size}')
        # pass SUL instead of the model, as the SUL handles the prediction logic of the RNN/transformer
        accuracy = test_accuracy_of_learned_classification_model(sul, learned_model, validation_data)

        if accuracy > 0.99 and not load:
            learned_model.save(f'learned_models/track_{TRACK}_dataset_{DATASET}_model')


            def predict(seq):
                return learned_model.execute_sequence(learned_model.initial_state, seq[1:-1])[-1]


            save_function(predict, nb_letters, f'dataset{TRACK}.{DATASET}', start_symbol, end_symbol,
                          submission_folder='submission_data')
