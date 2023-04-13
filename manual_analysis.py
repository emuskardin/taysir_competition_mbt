import pickle
import random
import sys
from collections import Counter, defaultdict
from random import randint, choices
from statistics import mean

import mlflow
import torch
from aalpy.base import SUL
from aalpy.learning_algs import run_KV, run_Lstar, run_RPNI, run_Alergia
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle
from aalpy.utils import convert_i_o_traces_for_RPNI
from aalpy.utils.HelperFunctions import all_prefixes

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
    not_solved_ids = [8, ]

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

        d = [list(zip(i[1:-1][:100],o[:100])) for i,o in validation_data_with_outputs.items()]
        d = convert_i_o_traces_for_RPNI(d)
        e = run_RPNI(d, automaton_type='dfa')
        for i in input_alphabet:
            if i not in e.initial_state.transitions.keys():
                e.initial_state.transitions[i] = e.initial_state

        e.make_input_complete('self_loop')

        num_correct = 0
        for inputs, outputs in validation_data_with_outputs.items():
            o = e.execute_sequence(e.initial_state, inputs[1:-1])[-1]
            if o == outputs[-1]:
                num_correct += 1

        print(num_correct / len(validation_data_with_outputs.keys()))
        e.save('rpni_model')
        exit()

        for inputs, outputs in validation_data_with_outputs.items():
            cuttoff = 1 - 0.31715267269232345
            if random.random() < cuttoff:
                pred = False
            else:
                pred = True
            if pred == outputs[-1]:
                num_correct +=1
            # p = all_prefixes(inputs[history_size:])
            # p.reverse()
            # for i in p:
            #     if i in history_based_prediction.keys():
            #         pred = choices(list(history_based_prediction[i].keys()), weights=list(history_based_prediction[i].values()))[0]
            #         if pred == outputs[-1]:
            #             num_correct += 1
            #         break

        print(num_correct / len(list(validation_data_with_outputs.values())))
        # exit()
        #
        # from aalpy.utils import convert_i_o_traces_for_RPNI
        #
        # io_traces = [list(zip(i[1:-1], o)) for i, o in validation_data_with_outputs.items()]
        # dataset = convert_i_o_traces_for_RPNI(io_traces)

        pos_seq, neg_seq = set(), set()

        character_map = Counter()
        for inputs, outputs in validation_data_with_outputs.items():
            inputs = inputs[1:-1]
            for index, o in enumerate(outputs):
                substring = inputs[:index + 1]
                if o:
                    pos_seq.add(substring)
                else:
                    neg_seq.add(substring)

        pos_seq, neg_seq = list(pos_seq), list(neg_seq)

        print(len(pos_seq) / len(neg_seq))

        print('pos', sum(sum(i) for i in pos_seq) / len(pos_seq))
        print('neg', sum(sum(i) for i in neg_seq) / len(neg_seq))

        exit()

        print(f'Num positive strings: {len(pos_seq)}')
        print(f'Num negative strings: {len(neg_seq)}')

        pos_str_lens = [s for s in pos_seq]
        pos_str_lens.sort(key=len)

        for i in range(20):
            print(pos_str_lens[i])

        pos_model = set(pos_seq)

        num_positive_outputs_random = 0
        for _ in range(1000):
            random_string = [start_symbol] + choices(input_alphabet, k=randint(100, 1000)) + [end_symbol]
            model_output = sul.get_model_output(random_string)
            output = tuple(random_string[1:-1]) in pos_model
            if output == model_output:
                num_positive_outputs_random += 1

        print(num_positive_outputs_random / 1000)

        exit()
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
                pos += 1
            if not outputs[-1] and neg < 10:
                print(f'False: {inputs[1:-1]}')
                neg += 1

        current_input = []
        while True:
            inp = int(input())
            if inp == -1:
                current_input = []
                continue
            current_input.append(inp)
            prediction = sul.get_model_output([start_symbol, ] + current_input + [end_symbol])
            print(f'Output for {current_input} : {prediction}')

        exit()
        # input_alphabet = [i[0] for i in freq_counter.most_common(n=3)]
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
