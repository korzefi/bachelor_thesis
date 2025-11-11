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
        self.encoder = nn.LSTM(input_dim, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        
        # Use the last output
        output = self.output_layer(encoder_outputs[:, -1, :])
        
        # Return dummy attention weights for compatibility
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        dummy_attention = torch.zeros(batch_size, seq_len)
        
        return output, dummy_attention
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


class AttentionRnnModel(nn.Module):
    """RNN model with attention mechanism."""
    
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, config: Dict[str, Any]) -> None:
        super(AttentionRnnModel, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = config.get('hidden_size', 256)
        self.dropout_rate = config.get('dropout', 0.2)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = nn.LSTM(input_dim, self.hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.dropout(input_seq)
        
        # Encoder
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        
        # Attention mechanism
        attention_scores = self.attention(encoder_outputs).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        context_vector = torch.sum(encoder_outputs * attention_weights.unsqueeze(-1), dim=1)
        
        # Output
        output = self.output_layer(context_vector)
        
        return output, attention_weights
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


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
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


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