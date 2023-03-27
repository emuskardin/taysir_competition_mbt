import pickle
import sys
from abc import ABC, abstractmethod
from statistics import mean

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import load_automaton_from_file
from aalpy.utils.HelperFunctions import all_prefixes

from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_regression_model, \
    load_validation_data_outputs, ValidationDataRegressionOracle, predict_transformer, visualize_density

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


class AbstractionMapper(ABC):

    @abstractmethod
    def to_abstract(self, x):
        pass

    @abstractmethod
    def to_concrete(self, x):
        pass


class NaiveBinPartitioning(AbstractionMapper):
    def __init__(self, validation_set_with_outputs, num_bins=100, consider_prefixes=True):
        observed_outputs = set()

        for _, outputs in validation_set_with_outputs.items():
            if consider_prefixes:
                observed_outputs.update(outputs)
            else:
                if outputs:
                    observed_outputs.add(outputs[-1])

        observed_outputs = list(observed_outputs)
        observed_outputs.sort()

        bin_size = len(observed_outputs) // num_bins

        bins = [observed_outputs[i:i + bin_size] for i in range(0, len(observed_outputs), bin_size)]
        self.bin_means = [mean(bin) for bin in bins]

    def to_abstract(self, x):
        diffs = []
        for bin_mean in self.bin_means:
            diffs.append(abs(x - bin_mean))

        return diffs.index(min(diffs))

    def to_concrete(self, x):
        return self.bin_means[x]


class RegressionRNNSUL(SUL):
    def __init__(self, rnn, mapper, is_transformer=False):
        super().__init__()
        self.rnn = rnn
        self.current_word = []
        self.mapper = mapper

        self.is_transformer = is_transformer

    def pre(self):
        self.current_word = []

    def post(self):
        pass

    def step(self, letter):
        if letter is not None:
            self.current_word.append(letter)
        prediction = self.get_model_output([start_symbol] + self.current_word + [end_symbol])

        if not self.mapper:
            return prediction

        return self.mapper.to_abstract(prediction)

    def get_model_output(self, seq):
        if self.is_transformer:
            return predict_transformer(self.rnn, seq)
        encoded_word = self.rnn.one_hot_encode(seq)
        return self.rnn.predict(encoded_word)

    def query(self, word: tuple) -> list:
        prediction = self.get_model_output((start_symbol,) + word + (end_symbol,))
        if not self.mapper:
            return prediction
        return [self.mapper.to_abstract(prediction)]


# disable all gradients
torch.set_grad_enabled(False)
# binary classification
track = 2
# all challenges
model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
current_test = [3]

for dataset in current_test:
    print(f'Track 2, Dataset {dataset}')
    model_name = f"models/{track}.{dataset}.taysir.model"

    model = mlflow.pytorch.load_model(model_name)
    model.eval()

    validation_data, start_symbol, end_symbol = get_validation_data(track, dataset)
    mean_input_seq_len = int(mean([len(l) for l in validation_data]))

    if dataset != 10:
        nb_letters = model.input_size - 1
        cell_type = model.cell_type
        print("The type of the recurrent cells is", cell_type.__name__)
    else:
        nb_letters = model.distilbert.config.vocab_size
        print("The model is a transformer (DistilBertForSequenceClassification)")
    print("The alphabet contains", nb_letters, "symbols.")

    sul = RegressionRNNSUL(model, None, is_transformer=dataset == 10)

    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

    mapper = NaiveBinPartitioning(validation_data_with_outputs, num_bins=10, consider_prefixes=False)
    sul.mapper = mapper

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=2,
                                   max_walk_len=10)
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)

    validation_oracle = ValidationDataRegressionOracle(input_alphabet, sul,
                                                       validation_data_with_outputs,
                                                       start_symbol, end_symbol, test_prefixes=False)

    # learned_model = run_Lstar(input_alphabet, sul, validation_oracle, 'moore',
    #                        max_learning_rounds=3,
    #                        cache_and_non_det_check=False)

    learned_model = load_automaton_from_file('ih_1.dot', 'moore')

    compact_model = learned_model.to_state_setup()

    for state_id, (output, transition_dict) in compact_model.items():
        compact_model[state_id] = (sul.mapper.to_concrete(output), transition_dict)

    compact_model['s0'] = (sul.get_model_output(tuple([start_symbol, end_symbol])), compact_model['s0'][1])

    test_accuracy_of_learned_regression_model(sul, compact_model, validation_data_with_outputs,
                                              num_random_sequances=100, random_seq_len=mean_input_seq_len // 1.5)

    with open(f'submission_data/pickles/model_{track}_{dataset}.pickle', 'wb') as handle:
        pickle.dump(compact_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # def predict(seq):
    #     current_state = 's0'
    #     for i in seq[1:-1]:
    #         current_state = compact_model[current_state][1][i]
    #     return compact_model[current_state][0]
    #
    #
    # save_function(predict, nb_letters, f'dataset{track}.{dataset}', start_symbol, end_symbol,
    #               submission_folder='submission_data')
