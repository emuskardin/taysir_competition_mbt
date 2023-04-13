from aalpy.base import SUL
from torch import IntTensor


class BinaryRNNSUL(SUL):
    def __init__(self, rnn, start_symbol, end_symbol):
        super().__init__()
        self.rnn = rnn
        self.current_word = []
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
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
        return self.get_model_output([self.start_symbol] + self.current_word + [self.end_symbol])
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

    def query(self, word: tuple) -> list:
        self.num_queries += 1
        self.num_steps += len(word)
        return [self.get_model_output((self.start_symbol,) + word + (self.end_symbol,))]


class BinaryTransformerSUL(SUL):
    def __init__(self, transformer, start_symbol, end_symbol):
        super().__init__()
        self.transformer = transformer
        self.current_word = []
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

    def pre(self):
        self.current_word = [self.start_symbol]

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            if not self.current_word:
                return self.get_model_output([self.start_symbol, self.end_symbol])
        else:
            self.current_word.append(letter)

        self.current_word.append(self.end_symbol)

        encoded_word = IntTensor([[1] + [a + 2 for a in self.current_word]])
        prediction = self.transformer(encoded_word)
        prediction = prediction.logits.argmax().item()

        self.current_word.pop(-1)

        return bool(prediction)

    def get_model_output(self, seq):
        encoded_word = IntTensor([[1] + [a + 2 for a in seq]])
        prediction = self.transformer(encoded_word)
        return bool(prediction.logits.argmax().item())
