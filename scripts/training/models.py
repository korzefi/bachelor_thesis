from scripts.training.config import NetParameters
from scripts.utils import get_root_path
import torch.nn
import torch.nn.functional as F

root_path = get_root_path()


class RnnModel(torch.nn.Module):
    """
    An RNN model using either LSTM
    """

    def __init__(self, seq_length, input_dim, output_dim):
        super(RnnModel, self).__init__()

        self.seq_length = seq_length
        self.hidden_size = NetParameters.hidden_size

        self.dropout = torch.nn.Dropout(NetParameters.dropout_p)

        self.encoder = torch.nn.LSTM(input_dim, self.hidden_size)

        self.out = torch.nn.Linear(self.hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        score_seq = self.out(encoder_outputs[-1, :, :])

        dummy_attn_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0])
        return score_seq, dummy_attn_weights  # No attention weights

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        return (h_init, c_init)

class AttentionRnnModel(torch.nn.Module):
    """
    An RNN model with classic attention mechanism
    """

    def __init__(self, seq_length, input_dim, output_dim):
        super(AttentionRnnModel, self).__init__()

        self.m = NetParameters.hidden_size
        self.T = seq_length
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(NetParameters.dropout_p)

        self.encoder = torch.nn.LSTM(input_dim, self.m)


        self.Uattn = torch.nn.Linear(self.m, self.m)
        self.vattn = torch.nn.Linear(self.m, seq_length)

        # decoder
        self.out = torch.nn.Linear(self.m, output_dim)

    def forward(self, x, hidden):
        x = self.dropout(x)
        encoder_output, (h_s, c_s) = self.encoder(x, hidden)

        attn_applied, weights = self.attention(encoder_output, h_s)
        score_seq = self.out(attn_applied.reshape(-1, self.m))

        return score_seq, weights

    def attention(self, encoder_outputs, h_s):
        # weights = F.softmax(torch.squeeze(self.attn(h_s)), dim=1)

        # attention auxiliary NNs
        e = self.vattn(torch.tanh((self.Uattn(h_s))))
        weights = F.softmax(torch.squeeze(e), dim=1)
        weights = torch.unsqueeze(weights, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(weights, encoder_outputs)

        return attn_applied, torch.squeeze(weights)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.m)
        c_init = torch.zeros(1, batch_size, self.m)

        return (h_init, c_init)


class DualAttentionRnnModel(torch.nn.Module):
    """
    A Dual-Attention RNN model, attending over both the input at each timestep
    and all hidden states of the encoder to make the final prediction.
    """

    def __init__(self, seq_length, input_dim, output_dim):
        super(DualAttentionRnnModel, self).__init__()

        self.n = input_dim
        self.m = NetParameters.hidden_size
        self.T = seq_length
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(NetParameters.dropout_p)

        self.encoder = torch.nn.LSTM(self.n, self.m)

        # input attn auxiliary NNs
        self.We = torch.nn.Linear(2 * self.m, self.T)
        self.Ue = torch.nn.Linear(self.T, self.T)
        self.ve = torch.nn.Linear(self.T, 1)

        # temporal attn auxiliary NNs
        self.Ud = torch.nn.Linear(self.m, self.m)
        self.vd = torch.nn.Linear(self.m, 1)

        # decoder
        self.out = torch.nn.Linear(self.m, output_dim)

    def forward(self, x, hidden_state):
        x = self.dropout(x)
        output_seq = []
        for t in range(self.T):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
            # LSTM here is built of one cell - x_tilde is only for time t
            output_t, hidden_state = self.encoder(x_tilde, hidden_state)
            output_seq.append(output_t)

        encoder_output = torch.cat(output_seq, dim=0)
        c, beta = self.temporal_attention(encoder_output)
        logits = self.out(c)

        return logits, torch.squeeze(beta)

    def input_attention(self, x, hidden_state, t):
        # (batch size, dim_vec, years)
        x = x.permute(1, 2, 0)
        h, c = hidden_state
        # size (batch size, 1, hidden units num)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        hc = torch.cat([h, c], dim=2)

        # Bahdenau formula for calculating the score for encoder
        e = self.ve(torch.tanh(self.We(hc) + self.Ue(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]

        x_tilde = alpha * xt

        return x_tilde, alpha

    def temporal_attention(self, encoder_output):
        encoder_output = encoder_output.permute(1, 0, 2)
        # Bahdenau formula for calculating the score for single decoder state
        l = self.vd(torch.tanh((self.Ud(encoder_output))))
        l = torch.squeeze(l)
        beta = F.softmax(l, dim=1)
        beta = torch.unsqueeze(beta, 1)
        c = torch.bmm(beta, encoder_output)
        c = torch.squeeze(c)

        return c, beta

    def init_hidden(self, batch_size):
        h_init = torch.zeros(1, batch_size, self.m)
        c_init = torch.zeros(1, batch_size, self.m)

        return (h_init, c_init)
