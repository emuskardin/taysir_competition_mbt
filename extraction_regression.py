import pickle
import sys
from abc import ABC, abstractmethod
from statistics import mean

import mlflow
import torch
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

from SULs import RegressionRNNSUL
from utils import get_validation_data, test_accuracy_of_learned_regression_model, \
    load_validation_data_outputs, ValidationDataRegressionOracle

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)


# Abstract class which can be used to create a new mapper
class AbstractionMapper(ABC):

    @abstractmethod
    def to_abstract(self, x):
        pass

    @abstractmethod
    def to_concrete(self, x):
        pass


# Mapper that partitions sorted outputs in predefined number of partitions
# Output is mapped to the partition to which it has minimal distance
class NaivePartitioning(AbstractionMapper):
    def __init__(self, validation_set_with_outputs, num_partitions=20, consider_prefixes=True):
        observed_outputs = set()

        for _, outputs in validation_set_with_outputs.items():
            if consider_prefixes:
                observed_outputs.update(outputs)
            else:
                if outputs:
                    observed_outputs.add(outputs[-1])

        observed_outputs = list(observed_outputs)
        observed_outputs.sort()

        partition_size = len(observed_outputs) // num_partitions

        partitions = [observed_outputs[i:i + partition_size] for i in range(0, len(observed_outputs), partition_size)]
        self.partition_means = [mean(p) for p in partitions]

    def to_abstract(self, x):
        diffs = []
        for partition_mean in self.partition_means:
            diffs.append(abs(x - partition_mean))

        return diffs.index(min(diffs))

    def to_concrete(self, x):
        return self.partition_means[x]


# disable all gradients
torch.set_grad_enabled(False)
# density estimation
track = 2
# all challenges
model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for dataset in model_ids:
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

    # Wrap a model in a SUL class
    sul = RegressionRNNSUL(model, None, start_symbol, end_symbol, is_transformer=dataset == 10)

    # define input alphabet
    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    # Load validation data and get start and end symbols
    validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

    # compute a mapper and assign it to the SUL
    mapper = NaivePartitioning(validation_data_with_outputs, num_partitions=20, consider_prefixes=True)
    sul.mapper = mapper

    # three different oracles, all can achieve same results depending on the parametrization
    # completely random oracle
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=20)
    # random oracle with state coverage
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=25, walk_len=15)
    # oracle that tests validation sequences (an additional oracle can be added to this oracle, currently set to None)
    validation_oracle = ValidationDataRegressionOracle(input_alphabet, sul,
                                                       validation_data_with_outputs,
                                                       start_symbol, end_symbol, test_prefixes=False)
    # run learning algorithms for 20 rounds
    learned_model = run_KV(input_alphabet, sul, validation_oracle,
                           automaton_type='moore',
                           max_learning_rounds=20,
                           cache_and_non_det_check=False)

    # replace discrete outputs in the learned model with partition mean
    for state in learned_model.states:
        state.output = sul.mapper.to_concrete(state.output)
    learned_model.initial_state.output = sul.get_model_output(tuple([start_symbol, end_symbol]))

    # to enable saving of large models due to pickle recursion limit
    compact_model = learned_model.to_state_setup()

    # test accuracy
    test_accuracy_of_learned_regression_model(sul, compact_model, validation_data_with_outputs,
                                              num_random_sequances=100, random_seq_len=mean_input_seq_len // 1.5)

    # save to pickle, used to create a submission for Codalabs
    with open(f'submission_data/pickles/model_{track}_{dataset}.pickle', 'wb') as handle:
        pickle.dump(compact_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
