#!/usr/bin/env python3
"""
Neural Network Architecture Definitions

This module contains the RNN model architectures for mutation prediction:
- RnnModel: Basic LSTM model
- AttentionRnnModel: LSTM with attention mechanism
- DualAttentionRnnModel: LSTM with dual attention (temporal + feature)
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnModel(nn.Module):
    """Basic RNN model using LSTM."""
    
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, config: Dict[str, Any]) -> None:
        super(RnnModel, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = config.get('hidden_size', 256)
        self.dropout_rate = config.get('dropout', 0.2)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = torch.nn.LSTM(input_dim, self.hidden_size)

        self.out = torch.nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        score_seq = self.out(encoder_outputs[-1, :, :])

        dummy_attn_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0], device=input_seq.device)
        return score_seq, dummy_attn_weights  # No attention weights

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))


class AttentionRnnModel(nn.Module):
    """RNN model with attention mechanism."""
    
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, config: Dict[str, Any]) -> None:
        super(AttentionRnnModel, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = config.get('hidden_size', 256)
        self.dropout_rate = config.get('dropout', 0.2)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = torch.nn.LSTM(input_dim, self.hidden_size)

        # attn auxiliary NNs
        self.Uattn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.vattn = torch.nn.Linear(self.hidden_size, seq_length)

        # decoder
        self.out = torch.nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(input_seq)
        encoder_output, (h_s, c_s) = self.encoder(x, hidden_state)

        attn_applied, weights = self.attention(encoder_output, h_s)
        score_seq = self.out(attn_applied.reshape(-1, self.hidden_size))

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

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))


class DualAttentionRnnModel(nn.Module):
    """RNN model with dual attention mechanism."""
    
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, config: Dict[str, Any]) -> None:
        super(DualAttentionRnnModel, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = config.get('hidden_size', 256)
        self.dropout_rate = config.get('dropout', 0.2)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = nn.LSTM(input_dim, self.hidden_size)
        
        # input attn auxiliary NNs
        self.We = torch.nn.Linear(2 * self.hidden_size, self.seq_length)
        self.Ue = torch.nn.Linear(self.seq_length, self.seq_length)
        self.ve = torch.nn.Linear(self.seq_length, 1)

        # temporal attn auxiliary NNs
        self.Ud = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.vd = torch.nn.Linear(self.hidden_size, 1)

        # decoder
        self.out = torch.nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(input_seq)
        output_seq = []
        for t in range(self.seq_length):
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
        x_tilde = torch.unsqueeze(x_tilde, 0)

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

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))


def create_model(model_type: str, seq_length: int, input_dim: int, output_dim: int, config: Dict[str, Any]) -> nn.Module:
    """Factory function to create models based on type."""
    model_classes = {
        'RNN': RnnModel,
        'AttentionRNN': AttentionRnnModel,
        'DualAttentionRNN': DualAttentionRnnModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    return model_class(seq_length, input_dim, output_dim, config)