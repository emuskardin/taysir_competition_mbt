import pickle
import sys
from statistics import mean

import mlflow
import torch
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

from SULs import BinaryRNNSUL, BinaryTransformerSUL
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_classification_model, \
    load_validation_data_outputs, test_sul_stepping

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)

# disable all gradients
torch.set_grad_enabled(False)
# binary classification
track = 1
# all challenges
model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# skip the first model as it is the biggest one
for dataset in model_ids[1:]:
    print(f'Track 1, Dataset {dataset}')

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

    # Load validation data and get start and end symbols
    validation_data, start_symbol, end_symbol = get_validation_data(track, dataset)
    val_data_mean_len = int(mean([len(x) - 2 for x in validation_data]))

    # Wrap a model in a SUL class
    if dataset != 7:
        sul = BinaryRNNSUL(model, start_symbol, end_symbol)
    else:
        sul = BinaryTransformerSUL(model, start_symbol, end_symbol)

    # define input alphabet
    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    # get outputs to strings in validation dataset
    validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

    # three different oracles, all can achieve same results depending on the parametrization
    # completely random oracle
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=1000,
                                   min_walk_len=10,
                                   max_walk_len=30,
                                   reset_after_cex=False)

    # random oracle with state coverage
    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=100, walk_len=val_data_mean_len)
    # oracle that tests validation sequences (an additional oracle can be added to this oracle, currently set to None)
    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, strong_eq_oracle,
                                                    validation_data_with_outputs,
                                                    start_symbol, end_symbol, test_prefixes=True)

    # run learning algorithms for 500 rounds
    # it will either terminate earlier, or in case of non-regular languages we will end up with 500 state model
    # (caching is not used due to implementation details of SUL)
    learned_model = run_KV(input_alphabet, sul, validation_oracle,
                           automaton_type='dfa',
                           max_learning_rounds=500,
                           cache_and_non_det_check=False)

    # visualize the learned model
    # learned_model.visualize()

    print(f'Testing model: Track 1, Model {dataset}: Model size {learned_model.size}')

    # to enable saving of large models due to pickle recursion limit
    compact_model = learned_model.to_state_setup()

    # test accuracy
    acc_val, acc_rand = test_accuracy_of_learned_classification_model(sul, compact_model,
                                                                      validation_data_with_outputs,
                                                                      num_random_sequances=1000,
                                                                      random_seq_len=val_data_mean_len // 1.5)
    # save to pickle, used to create a submission for Codalabs
    with open(f'submission_data/pickles/model_{track}_{dataset}.pickle', 'wb') as handle:
        pickle.dump(compact_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
