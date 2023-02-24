"""tnetwork

This module implements a neural network as a PyTorch Module.
"""

__all__ = ['TNetwork', 'pad', 'unbatch', 'unpad']

import torch
from torch import nn

def fqcn(c):
    m = c.__module__
    if m == 'builtins':
        return c.__qualname__
    return m + '.' + c.__qualname__

class TNetwork(nn.Module):

    ### public:

    def __init__(self, input_size, n_outputs, *,
                 dtype=None, device=torch.device('cpu'),
                 activation=None, **kwargs):
        super().__init__()
        # Available hyperparameters and their default values
        self.n_layers  = 1
        self.neurons_per_layer = 32
        self.dropout = 0
        self.cell_type = nn.RNN
        self.final_type = nn.Linear
        self.batch_size = 32
        self.patience = None
        self.bidirectional = False
        self.split_dense = False
        self.task = "binary"

        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)
        # End of hyperparameter handling
        self.input_size = input_size
        self.n_outputs = n_outputs

        bidimul = 2 if self.bidirectional else 1

        # we need to know whether hidden state is a single tensor
        # like for RNN / GRU, or a pair of tensors like for LSTM.
        # Rather than detecting specific types for this information,
        # we should extract it directly:
        self.hides_pairs = type(self.cell_type(1,1,1)
                                (torch.zeros(1,1))[1]) is tuple

        # in order to expose as much of the hidden state as possible,
        # stacked recurrent layers are created manually.
        # the module's input_size includes a padding symbol;
        # padding is NEVER passed to the layers that make up the module,
        # and so is excluded from these machines
        self.mach = []
        self.mach.append(self.cell_type(self.input_size-1,
                                        self.neurons_per_layer,
                                        1,
                                        dtype=dtype,
                                        device=device,
                                        dropout=self.dropout,
                                        batch_first=True,
                                        bidirectional=self.bidirectional))
        for i in range(self.n_layers - 1):
            self.mach.append(self.cell_type(self.neurons_per_layer
                                            * bidimul,
                                            self.neurons_per_layer
                                            * bidimul,
                                            1,
                                            dtype=dtype,
                                            device=device,
                                            dropout=self.dropout,
                                            batch_first=True,
                                            bidirectional=False))
        for i,m in enumerate(self.mach):
            self.add_module("mach["+str(i)+"]", m)
        layers = []
        # Always match output size to input size of next layer.
        # Add a dropout layer
        if self.dropout:
            layers.append(nn.Dropout(self.dropout,
                                     dtype=dtype, device=device))
        # Add a fully-connected layer
        out_features = self.n_outputs
        if self.split_dense:
            out_features = self.neurons_per_layer // 2
        in_features = self.neurons_per_layer * bidimul
        layer_type = nn.Linear if self.split_dense else self.final_type
        layers.append(layer_type(in_features, out_features,
                                 dtype=dtype,device=device))
        # Possibly add another fully-connected layer
        if self.split_dense:
            out_features = self.n_outputs
            in_features = self.neurons_per_layer // 2
            layers.append(self.final_type(in_features, out_features,
                                          dtype=dtype,
                                          device=device))
        # Add an activation layer
        if activation is not None:
            layers.append(activation.to(dtype=dtype,
                                        device=device))
        layers.append(nn.Softmax(dim=-1))
        self.dense = torch.nn.Sequential(*layers)

    def extra_repr(self) -> str:
        s = '{input_size}, {n_outputs}'
        if self.n_layers != 1:
            s += ', n_layers={n_layers}'
        if self.neurons_per_layer != 32:
            s += ', neurons_per_layer={neurons_per_layer}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.cell_type != nn.RNN:
            s += f', cell_type={fqcn(self.cell_type)}'
        if self.final_type != nn.Linear:
            s += ', final_type={final_type}'
        if self.batch_size != 32:
            s += ', batch_size={batch_size}'
        if self.patience is not None:
            s += ', patience={patience}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        if self.split_dense is not False:
            s += ', split_dense={split_dense}'
        if self.task != "binary":
            s += ', task={task}'
        return s.format(**self.__dict__)

    def feed_dense(self, x):
        return self.dense(x)

    def forward(self, x, hidden=None, full_ret=False):
        if self.task == "lm":
            return self.forward_lm(x, hidden, full_ret)
        return self.forward_bin(x, hidden, full_ret)

    def forward_bin(self, x, hidden=None, full_ret=False):
        words = unpad(unbatch(x))
        states = [self._pass_recurrent(word, hidden, b=b)
                  for b,word in enumerate(words)]
        outs, hidden = self._interpret_states(states)
        fed = self.feed_dense(torch.stack(outs))
        if x.dim() == 2:
            fed, hidden = self._squeezeoh(fed, hidden)
        if full_ret:
            all_s = [[layer[-1] for layer in state] for state in states]
            return fed, hidden, all_s
        return fed, hidden

    def forward_lm(self, x, hidden=None, full_ret=False):
        # if full_ret is True:
        # the return is a triple (out, hidden, ah)
        # `ah` is a list whose length is the number of input words.
        # per word, the corresponding element is another list
        # whose length is the length of that input word ---
        # as `x` is unbatched, these lengths may not be comparable.
        # within THAT, each symbol gets a list of length self.n_layers
        # containing a hidden state for that layer.
        # tl;dr: (batch_size, word_length, n_layers, state)
        words = unbatch(x)
        is_batched = x.dim() != 2
        all_outs = []
        all_h = []
        all_state = []
        ubh = self._split_hidden(hidden)
        if ubh is None:
            ubh = len(words) * [None]
        for word,h_0 in zip(words,ubh):
            out = [word[0][(0 if is_batched else 1):]]
            size = unpad((word,))[0].size(0)
            word_h = []
            for i in [i + 1 for i in range(size - 1)]:
                ohx = self.forward_bin(word[:i], h_0, full_ret)
                o, h = ohx[:2]
                if is_batched:
                    z1 = torch.tensor([0], dtype=o.dtype, device=o.device)
                    o = torch.cat((z1,o))
                out.append(o)
            ohx = self.forward_bin(word, h, full_ret)
            all_outs.append(torch.stack(out))
            all_h.append(ohx[1])
            word_h.append(ohx[-1][0])
            all_state.append(word_h)
        max_len = x.size(-2)
        all_outs = torch.stack([pad(o,max_len) for o in all_outs])
        hidden = self._interpret_lm_states(all_h)
        if not is_batched:
            all_outs, hidden = self._squeezeoh(all_outs, hidden)
        if full_ret:
            return all_outs, hidden, all_state
        return all_outs, hidden

    def one_hot_encode(self, word):
        tensor=torch.zeros(len(word),self.input_size,
                           dtype=self.dtype,device=self.device)
        for i,symbol in enumerate(word):
            tensor[i,symbol+1] = 1
        return tensor

    def predict(self, x):
        with torch.no_grad():
            if self.task == 'lm':
                return self.predict_lm(x)
            return self.predict_bin(x)

    def predict_bin(self, x):
        out, _ = self(x)
        out = out.argmax(-1).flatten()
        out = out >= 0.5
        if x.dim() == 2:
            return float(out)
        return out.float()

    def predict_lm(self, x):
        if self.input_size != self.n_outputs + 1:
            raise ValueError('I/O size mismatch.  Input wants '
                             f'{self.input_size} but output wants '
                             f'{self.n_outputs}. The former should '
                             'be one greater than the latter.')
        if x.size(-1) != self.input_size:
            if x.size(-1) != self.input_size - 1:
                raise ValueError(f'Need {self.input_size} input features '
                                 f'but received {x.size(-1)}')
            z = torch.zeros(*x.size()[:-1],1,
                            dtype=self.dtype,device=self.device)
            x = torch.cat((z,x),-1)
        t = self.task
        out, _ = self.forward_lm(x)
        likelihoods = []
        for word,val in zip(unbatch(x),unbatch(out)):
            word = unpad([word])[0]
            val = val[:word.size(-2)]
            if word.size(-1) != val.size(-1):
                val = unpad([val])[0]
            probs = word * val # pointwise-multiply
            probs = probs.sum(-1)
            probs = probs.prod()
            likelihoods.append(probs)
        likelihoods = torch.stack(likelihoods)
        if x.dim() == 2:
            return float(likelihoods)
        return likelihoods.float()

    def reached_hidden(self, x, hidden=None):
        with torch.no_grad():
            return self.forward(x, hidden, full_ret=True)[-1]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    ### private:

    def _interpret_states(self, states):
        outs = [state[-1][1] for state in states]
        if len(outs) == 0:
            return [], hidden
        states = [[layer[1] for layer in wordout] for wordout in states]
        states = list(map(list,zip(*states)))
        if self.hides_pairs:
            outs = [v[0] for v in outs]
            hidden = ([torch.stack([x[0] for x in state],1)
                       for state in states],
                      [torch.stack([x[1] for x in state],1)
                       for state in states],
                      )
            hidden = list(zip(*hidden))
        else:
            hidden = [torch.stack(h,1) for h in states]
        outs = [v.flatten() for v in outs]
        return outs, hidden

    def _interpret_lm_states(self, states):
        states = list(map(list,zip(*states)))
        if self.hides_pairs:
            hidden = ([torch.stack([x[0] for x in state],1)
                       for state in states],
                      [torch.stack([x[1] for x in state],1)
                       for state in states],
                      )
            hidden = list(zip(*hidden))
        else:
            hidden = [torch.stack(h,1) for h in states]
        return hidden

    def _pass_recurrent(self, word, hidden=None, b=0):
        """
        INPUTS:
            word: a 2D tensor shaped (L, input_size-1)
            hidden: an optional list of initial hidden(/cell) states
        OUTPUTS:
            a list of length self.num_layers containing tuples.
            each tuple contains three elements.
            the first is the hidden state at each time-step of that layer.
            the second is the final hidden(/cell)-state for that layer
            (if bidirectional, this is accounted for).
            the third is ALL hidden(/cell)-states,
            either identical to the first, or, for LSTM,
            in a special format.
        """
        if hidden is None:
            hidden = len(self.mach) * [None]
        options = dict()
        if hasattr(self.cell_type,'forward_x'):
            options['full_ret'] = True
        out = (word,)
        outputs = []
        for i,machine in enumerate(self.mach):
            h = hidden[i]
            if h is not None:
                if self.hides_pairs and h[0].dim() == 3:
                    h = [t[:,b] for t in h]
                elif not self.hides_pairs and h.dim() == 3:
                    h = h[:,b]
            out = machine(out[0], h, **options)
            if len(out) != 3:
                out = (*out, out[0])
            outputs.append(out)
        return outputs

    def _split_hidden(self, h):
        """
        the hidden state is passed as a list of length self.n_layers
        each element of which is a tensor (for RNN/GRU) or a pair of
        tensors (for LSTM).  these tensors are 2D for unbatched input,
        in which case they are already trivially "split",
        otherwise they are 3D with the "batch" dimension being the
        second.

        this function takes this n_layers-length list and returns
        a batch_size-long list of similar lists containing the 2D
        sort of tensor(/pair).

        if the hidden state is None, this indicates the use of defaults
        (usually zeros) and is returned entirely untouched.
        """
        if h is None:
            return h
        if len(h) == 0:
            return None
        if self.hides_pairs:
            if h[0][0].dim() == 2:
                return (h,)
            states = [torch.stack(layer).unbind(-2) for layer in h]
            states = list(map(list,zip(*states)))
            states = [[layer.unbind(0) for layer in b] for b in states]
            return states
        if h[0].dim() == 2:
            return (h,)
        states = [layer.unbind(-2) for layer in h]
        states = list(map(list,zip(*states)))
        return states

    def _squeezeoh(self,o,h):
        o = o.squeeze(0)
        if self.hides_pairs:
            h = [(hx.squeeze(1), cx.squeeze(1)) for hx,cx in h]
        else:
            h = [hx.squeeze(1) for hx in h]
        return o,h

def pad(x, max_len):
    s = x.size(-2)
    if s == max_len:
        return x
    if s > max_len:
        raise ValueError('Sequence too long.  Should be at most'
                         f' {max_len} but is actually {x.size(-2)}.')
    z = torch.zeros(*x.size()[:-2], max_len-s, x.size(-1),
                    dtype=x.dtype, device=x.device)
    z[...,0] = 1
    return torch.cat((x,z),-2)

def unbatch(x):
    """
    Split a 3D tensor whose first dimension is the batch_size
    into a tuple of 2D tensors representing its elements.

    INPUT:
        x: a 2D or 3D tensor
    OUTPUT:
        a tuple of 2D tensors
    """
    if x.dim() == 2:
        return (x,)
    return x.unbind()

def unpad(x):
    """
    Strip padding from a batch of inputs.

    INPUT:
        x: an iterable container of 2D tensors
    OUTPUT:
        a tuple of 2D tensors corresponding to `x`
        but with each tensor truncated to its pre-padding portion
        and without a slot for a padding symbol
    """
    # we want a contain of 2D tensors, not a single tensor!
    if type(x) is torch.Tensor:
        x = unbatch(x)
    words = []
    for word in x:
        t = 0
        unbroken = True
        for t,sym in enumerate(word):
            if int(sym.argmax()) == 0:
                unbroken = False
                break
        if unbroken:
            t = len(word)
        words.append(word[:t,1:])
    return tuple(words)
