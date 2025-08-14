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
        self.encoder = nn.LSTM(input_dim, self.hidden_size, batch_first=True)
        
        # Dual attention mechanism
        self.temporal_attention = nn.Linear(self.hidden_size, 1)
        self.feature_attention = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.output_layer = nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_seq: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.dropout(input_seq)
        
        # Encoder
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        
        # Temporal attention
        temporal_scores = self.temporal_attention(encoder_outputs).squeeze(-1)
        temporal_weights = F.softmax(temporal_scores, dim=1)
        
        # Feature attention
        feature_weights = torch.sigmoid(self.feature_attention(encoder_outputs))
        
        # Apply both attentions
        attended_features = encoder_outputs * feature_weights
        context_vector = torch.sum(attended_features * temporal_weights.unsqueeze(-1), dim=1)
        
        # Output
        output = self.output_layer(context_vector)
        
        return output, temporal_weights
    
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