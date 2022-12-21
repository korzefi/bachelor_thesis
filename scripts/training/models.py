from scripts.training.config import NetParameters
from scripts.utils import get_root_path
import torch.nn
import torch.nn.functional as F

root_path = get_root_path()

class AttentionRnnModel(torch.nn.Module):
    """
    An RNN model with classic attention mechanism,
    attending over both the input at each timestep
    and all hidden states of the encoder to make the final prediction.
    """

    def __init__(self, seq_length, input_dim, output_dim):
        super(AttentionRnnModel, self).__init__()

        self.m = NetParameters.hidden_size
        self.T = seq_length
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(NetParameters.dropout_p)

        self.encoder = torch.nn.LSTM(input_dim, self.m)

        self.attn = torch.nn.Linear(self.m, seq_length)

        self.out = torch.nn.Linear(self.m, output_dim)

    def forward(self, x, hidden):
        x = self.dropout(x)
        encoder_output, (h_s, c_s) = self.encoder(x, hidden)

        attn_applied, weights = self.attention(encoder_output, h_s)
        score_seq = self.out(attn_applied.reshape(-1, self.m))

        return score_seq, weights

    def attention(self, encoder_outputs, h_s):
        weights = F.softmax(torch.squeeze(self.attn(h_s)), dim=1)
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

        self.We = torch.nn.Linear(2 * self.m, self.T)
        self.Ue = torch.nn.Linear(self.T, self.T)
        self.ve = torch.nn.Linear(self.T, 1)

        self.Ud = torch.nn.Linear(self.m, self.m)
        self.vd = torch.nn.Linear(self.m, 1)
        self.out = torch.nn.Linear(self.m, output_dim)

    def forward(self, x, hidden_state):
        x = self.dropout(x)
        output_seq = []
        for t in range(self.T):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
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

        e = self.ve(torch.tanh(self.We(hc) + self.Ue(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]

        x_tilde = alpha * xt
        x_tilde = torch.unsqueeze(x_tilde, 0)

        return x_tilde, alpha

    def temporal_attention(self, encoder_output):
        encoder_output = encoder_output.permute(1, 0, 2)
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

