"""lstmx
an LSTM, eXtendend to output intermediate cell states
"""

import numbers
import warnings

import torch
from torch import nn

__all__ = ["LSTMx"]


class LSTMx(torch.nn.Module):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN
    to an input sequence.  This documentation has been copied
    with slight modification from the standard torch.nn.LSTM
    module.

    For each element in the input sequence, each layer computes
    the following function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{t-1} + b_{hi})\\
            f_t = \sigma(W_{if}x_t + b_{if} + W_{hf}h_{t-1} + b_{hf})\\
            g_t = \tanh (W_{ig}x_t + b_{ig} + W_{hg}h_{t-1} + b_{hg})\\
            o_t = \sigma(W_{io}x_t + b_{io} + W_{ho}h_{t-1} + b_{ho})\\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t\\
            h_t = o_t \odot \tanh(c_t)\\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`,
    :math:`c_t` is the cell state at time `t`,
    :math:`x_t` is the input at time `t`,
    :math:`h_{t-1}` is the hidden state of the layer at time `t-1`
    or the initial hidden state at time `0`, and
    :math:`i_t`, :math:`f_t`, :math:`g_t`, and :math:`o_t`
    are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and
    :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t`
    of the :math:`l`-th layer (:math:`l >= 2`) is the hidden state
    :math:`h^{(l-1)}_t` of the previous layer
    multiplied by dropout :math:`\delta^{(l-1)}_t`
    where each :math:`\delta^{(l-1)}_t` is a Bernoulli random variable
    which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers.
            E.g., setting ``num_layers=2`` would mean stacking
            two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first
            LSTM and computing the final results.  Default: 1
        bias: If ``False``, then the layer does not use bias weights
            `b_ih` and `b_hh`.  Default: ``True``
        batch_first: If ``True``, then the input and output tensors
            are provided as `(batch, seq, feature)` instead of
            `(seq, batch, feature)`.  Note that this does not apply
            to hidden or cell states.  See the Inputs/Outputs
            sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the
            outputs of each LSTM layer except the last layer,
            with dropout probability equal to :attr:`dropout`.
            Default:  0
        bidirectional: If ``True``, becomes a bidirectional LSTM.
            Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections
            of corresponding size.  Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape
          :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False``, or
          :math:`(N, L, H_{in})` when ``batch_first=True``
          containing the features of the input sequence.
        * **h_0**: tensor of shape
          :math:`(D*\text{num\_layers}, H_{out})` for unbatched input
          or :math:`(D*\text{num\_layers}, N, H_{out})`
          containing the initial hidden state for each element
          in the input sequence.
          Defaults to zeros of (h_0, c_0) is not provided.
        * **c_0**: tensor of shape
          :math:`(D*\text{num\_layers}, H_{out})` for unbatched input
          or :math:`(D*\text{num\_layers}, N, H_{out})`
          containing the initial cell state for each element
          in the input sequence.
          Defaults to zeros of (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size}\\
                L ={} & \text{sequence length}\\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1\\
                H_{in} ={} & \text{input\_size}\\
                H_{cell} ={} & \text{hidden\_size}\\
                H_{out} ={} & \text{proj\_size}
                    \text{ if proj_size}>0
                    \text{ otherwise hidden\_size}\\
            \end{aligned}

    Outputs: (output,all_c), (h_n, c_n)
        * **output**: tensor of shape
          :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`.
          When ``bidirectional=True``, `output` will contain
          a concatenation of the forward and reverse hidden states
          at each time step in the sequence.
        * **all_c**: tensor of shape
          :math:`(L, D * \text{num\_layers}, H_{cell})`
          for unbatched input, or
          :math:`(L, D * \text{num\_layers, N, H_cell})`
          containing the cell state
          `(c_t)` from the last layer of the LSTM, for each `t`.
          When ``bidirectional=True``, `output` will contain
          a concatenation of the forward and reverse hidden states
          at each time step in the sequence.
        * **h_n**: tensor of shape
          :math:`(D * \text{num\_layers}, H_{cell})`
          for unbatched input, or
          :math:`(D * \text{num\_layers, N, H_cell})`
          containing the final hidden state for each element
          in the sequence.  This is the final plane of ``output``
          when ``bidirectional=False``, or the relevant parts
          from the final and initial planes.
        * **c_n** tensor of shape
          :math:`(D * \text{num\_layers}, H_{cell})`
          for unbatched input, or
          :math:`(D * \text{num\_layers, N, H_cell})`
          containing the final cell state for each element
          in the sequence.  This is the final plane of ``all_c``
          when ``bidirectional=False``, or the relevant parts
          from the final and initial planes.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the
            :math:`\text{k}^{th}` layer `(W_ii|W_if|W_ig|W_io)`,
            of shape `(4*hidden_size, input_size)` for `k=0`.
            Otherwise, the shape is
            `(4*hidden_size, num_directions * hidden_size)`.
            If ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        weight_hh_l[k] : the learnable hidden-hidden weights of the
            :math:`\text{k}^{th}` layer `(W_hi|W_hf|W_hg|W_ho)`,
            of shape `(4*hidden_size, input_size)` for `k=0`.
            Otherwise, the shape is
            `(4*hidden_size, num_directions * hidden_size)`.
            If ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        bias_ih_l[k] : the learnable input-hidden bias of the
            :math:`\text{k}^{th}` layer `(b_ii|b_if|b_ig|b_io)`,
            of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the
            :math:`\text{k}^{th}` layer `(b_hi|b_hf|b_hg|b_ho)`,
            of shape `(4*hidden_size)`
        weight_ih_l[k]_reverse : Analogous to `weight_ih_l[k]`
            for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hh_l[k]_reverse : Analogous to `weight_hh_l[k]`
            for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_ih_l[k]_reverse : Analogous to `bias_ih_l[k]`
            for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_hh_l[k]_reverse : Analogous to `bias_hh_l[k]`
            for the reverse direction.
            Only present when ``bidirectional=True``.

    .. note::
        All the weights and biases are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional LSTMs, forward and backward
        are directions 0 and 1 respectively.  Example of splitting
        the ouput layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        For bidirectional LSTMs, `h_n` is not equivalent to the
        last element of `output`.  The former contains the final
        forward and reverse hidden states, while the latter
        contains the final forward hidden state and the initial
        reverse hidden state.

    .. note::
      ``batch_first`` argument is ignored for unbatched inputs.

    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = LSTMx(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> (output, all_c), (hn, cn) = rnn(input, (h0, c0))
    """
    __constants__ = [
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ]
    __jit_unused_properties__ = ["all_weights"]

    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMx, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0,1] "
                "representing the probability of an element "
                "being zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all "
                "but last recurrent layer, so non-zero "
                "dropout expects num_layers greater than 1, "
                "but got dropout={} and num_layers={}"
                "".format(dropout, num_layers)
            )
        self.drop_layer = torch.nn.Dropout(dropout)
        self.forward_layers = []
        for layer in range(num_layers):
            i = input_size if layer == 0 else hidden_size * num_directions
            self.forward_layers.append(torch.nn.LSTMCell(i, hidden_size))
        for i, m in enumerate(self.forward_layers):
            self.add_module("forward_layers[" + str(i) + "]", m)
        self.reverse_layers = None

        if not bidirectional:
            return
        self.reverse_layers = []
        for layer in range(num_layers):
            i = input_size if layer == 0 else hidden_size * num_directions
            self.reverse_layers.append(torch.nn.LSTMCell(i, hidden_size))
        for i, m in enumerate(self.reverse_layers):
            self.add_module("reverse_layers[" + str(i) + "]", m)

    def extra_repr(self):
        """
        Return extra information used in printing the module.
        """
        s = "{input_size}, {hidden_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def forward(self, x, hxcx=None, full_ret=None):
        out = self.forward_x(x, hxcx)
        if full_ret:
            return out
        return out[0], out[1]

    def forward_x(self, x, hxcx=None):
        """
        inputs: x, (hx, cx)
            x : input of shape (N, L, H_{in}) if batch_first==True
                else of shape (L, N, H_{in})
            hx: initial hidden state, of shape (D*num_layers, N, H_out)
            cx: initial cell state, of shape (D*num_layers, N, H_out)
        outputs: out, (hn, cn), (all_h, all_c)
            out  : output of shape (N, L, D*H_{out}) if batch_first==True
                   else of shape(L, N, D*H_{out}).
                   If self.bidirectional==True, output will contain
                   a concatenation of the forward
                   and reverse hidden states at each time step
                   in the sequence.
                   This is the output from the final layer.
            hn   : final hidden state for each element in the sequence,
                   shaped (D*num_layers, H_out) if unbatched or
                   (D*num_layers, N, H_out).  If birectional, this
                   is the concatenation of the final time step
                   for each direction.
            cn   : final cell state for each element in the sequence,
                   shaped (D*num_layers, H_out) if unbatched or
                   (D*num_layers, N, H_out).  If birectional, this
                   is the concatenation of the final time step
                   for each direction.
            all_h: shape is (num_layers, N, L, D*H_{out})
                   if batch_first==True
                   else (num_layers, L, N, D*H_{out})
            all_c: shape is (num_layers, N, L, D*H_{out})
                   if batch_first==True
                   else (num_layers, L, N, D*H_{out})
        """
        num_directions = 2 if self.bidirectional else 1
        batch_size = x.size(0) if self.batch_first else x.size(1)
        if x.dim() != 3:
            batch_size = 1
        if hxcx is None:
            hx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
            cx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
            hxcx = hx, cx
        if x.dim() == 3:
            order = (1, 0, 2) if self.batch_first else (0, 1, 2)
            batched = torch.permute(x, order)
        else:
            batched = x

        hx, cx = hxcx
        hx = hx.view(num_directions, self.num_layers, batch_size, self.hidden_size)
        cx = cx.view(num_directions, self.num_layers, batch_size, self.hidden_size)
        if x.dim() == 2:
            hx = hx.squeeze(2)
            cx = cx.squeeze(2)
        all_h = []
        all_c = []
        for i, _ in enumerate(self.forward_layers):
            if i != 0:
                batched = self.drop_layer(batched)
            h, c = self._run_layer(batched, hx[:, i], cx[:, i], i)
            batched = h
            all_h.append(h)
            all_c.append(c)
        all_h, all_c = torch.stack(all_h), torch.stack(all_c)
        # all_[hc] are shaped (num_layers, L, N, D*h_out)
        final_out = h
        h_n = all_h[:, -1, ..., : self.hidden_size]
        c_n = all_c[:, -1, ..., : self.hidden_size]
        if self.bidirectional:
            h_n = torch.cat([h_n, all_h[:, 0, ..., self.hidden_size :]], -1)
            h_n = h_n.view(2 * self.num_layers, -1, self.hidden_size)
            c_n = torch.cat([c_n, all_h[:, 0, ..., self.hidden_size :]], -1)
            c_n = c_n.view(2 * self.num_layers, -1, self.hidden_size)
        if x.dim() == 2:
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1)
        if self.batch_first:
            if all_h.dim() == 4:
                all_h = torch.permute(all_h, (0, 2, 1, 3))
                all_c = torch.permute(all_c, (0, 2, 1, 3))
                final_out = torch.permute(final_out, (1, 0, 2))
        return final_out, (h_n, c_n), (all_h, all_c)

    def _run_layer(self, x, hx, cx, t):
        """
        Inputs: x, hx, cx, t
            * x  :: (L, N, H_in)  or (L, H_in)
            * hx :: (D, N, H_out) or (D, H_out)
            * cx :: (D, N, H_out) or (D, H_out)
            * t  :: int
        Outputs: (outs, cells)
            * outs  :: (L, N, D * H_out) or (L, D * H_out)
            * cells :: (L, N, D * H_out) or (L, D * H_out)
        """
        timesteps = x.unbind(0)
        h_out, c_out = [], []
        h, c = hx[0], cx[0]
        for element in timesteps:
            h, c = self.forward_layers[t](element, (h, c))
            h_out.append(h)
            c_out.append(c)
        if not self.bidirectional:
            return torch.stack(h_out), torch.stack(c_out)
        h, c = hx[1], cx[1]
        for i, element in enumerate(reversed(timesteps)):
            h, c = self.reverse_layers[t](element, (h, c))
            h_out[-i - 1] = torch.cat([h_out[-i - 1], h], -1)
            c_out[-i - 1] = torch.cat([c_out[-i - 1], c], -1)
        return torch.stack(h_out), torch.stack(c_out)
