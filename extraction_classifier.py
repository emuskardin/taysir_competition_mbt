import pickle
import sys
from statistics import mean

import mlflow
import torch
from aalpy.learning_algs import run_KV, run_Lstar
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle

from SULs import BinaryRNNSUL, BinaryTransformerSUL
from submit_tools import save_function
from utils import get_validation_data, ValidationDataOracleWrapper, test_accuracy_of_learned_classification_model, \
    load_validation_data_outputs, test_sul_stepping

print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
print("Your python version:", sys.version)

# Current results (13.03)
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


# disable all gradients
torch.set_grad_enabled(False)
# binary classification
track = 1
# all challenges
model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# helper
perfect = [2, 3, 4, 5, 7]
almost_perfect = [1, 6, 9, 10]
not_solved_ids = [8, 11]

to_sample = [11]
current_test = [4]

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
    val_data_mean_len = int(mean([len(x) - 2 for x in validation_data]))

    if dataset != 7:
        sul = BinaryRNNSUL(model, start_symbol, end_symbol)
    else:
        sul = BinaryTransformerSUL(model, start_symbol, end_symbol)

    input_alphabet = list(range(nb_letters))
    input_alphabet.remove(start_symbol)
    input_alphabet.remove(end_symbol)

    validation_data_with_outputs = load_validation_data_outputs(sul, validation_data, track, dataset)

    # three different oracles, all can achieve same results depending on the parametrization
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=1000,
                                   min_walk_len=10,
                                   max_walk_len=30,
                                   reset_after_cex=False)

    strong_eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=100, walk_len=val_data_mean_len)

    validation_oracle = ValidationDataOracleWrapper(input_alphabet, sul, None,
                                                    validation_data_with_outputs,
                                                    start_symbol, end_symbol, test_prefixes=True)

    learned_model = run_Lstar(input_alphabet, sul, validation_oracle,
                              automaton_type='dfa',
                              max_learning_rounds=None,
                              cache_and_non_det_check=False)

    print(f'Testing model: Track 1, Model {dataset}: Model size {learned_model.size}')

    learned_model.initial_state.transitions[start_symbol] = learned_model.initial_state
    learned_model.initial_state.transitions[end_symbol] = learned_model.initial_state
    learned_model.make_input_complete('self_loop')

    # to enable saving of large models due to pickle recursion limit
    compact_model = learned_model.to_state_setup()

    # test accuracy
    acc_val, acc_rand = test_accuracy_of_learned_classification_model(sul, compact_model,
                                                                      validation_data_with_outputs,
                                                                      num_random_sequances=1000,
                                                                      random_seq_len=val_data_mean_len // 1.5)

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
