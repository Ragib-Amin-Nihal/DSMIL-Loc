import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create a larger buffer to accommodate potentially larger dimensions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Dynamically adjust positional encoding to match input dimensions
        target_size = x.size(-1)
        
        # If target size is larger than current encoding, repeat or extend
        if target_size > self.d_model:
            # Repeat the encoding to match the target size
            repeats = (target_size + self.d_model - 1) // self.d_model
            pe = self.pe.repeat(1, 1, repeats)[:, :x.size(1), :target_size]
        else:
            # Truncate the encoding if target size is smaller
            pe = self.pe[:, :x.size(1), :target_size]
        
        return x + pe

class GateModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.gate(x)

class ImprovedLocalizationMILNet(nn.Module):

    def __init__(self, feature_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        # Increased embedding dimension to match combined features
        COMBINED_DIM = 1024
        
        # Improved Spectrogram Encoder with residual connections
        self.spec_encoder = nn.ModuleList([
            ResidualBlock(1, 32),      # [B, 32, F, T]
            ResidualBlock(32, 64),     # [B, 64, F/2, T/2]
            ResidualBlock(64, 128),    # [B, 128, F/4, T/4]
            ResidualBlock(128, 256)    # [B, 256, F/8, T/8]
        ])
        
        # Global context module
        self.context_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # Temporal feature encoder with more capacity
        self.temporal_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # Multi-head self-attention with matching dimension
        self.self_attention = nn.MultiheadAttention(
            embed_dim=COMBINED_DIM,  # Match the combined features dimension
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(COMBINED_DIM)
        
        # Instance scoring with gating - adjust for new dimension
        self.instance_scorer = nn.Sequential(
            nn.Linear(COMBINED_DIM * 2, COMBINED_DIM),
            nn.LayerNorm(COMBINED_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(COMBINED_DIM, COMBINED_DIM),
            nn.LayerNorm(COMBINED_DIM),
            GateModule(COMBINED_DIM),
            nn.Linear(COMBINED_DIM, 1)
        )
        
        # Bag classifier with multiple heads - adjust for new dimension
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(COMBINED_DIM, COMBINED_DIM),
                nn.LayerNorm(COMBINED_DIM),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(COMBINED_DIM, 1)
            ) for _ in range(num_heads)
        ])
        
        # Trainable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, spectrograms, features, num_instances):
        # Fix dimensionality checking for spectrograms
        if len(spectrograms.shape) == 4:
            # If input is [batch_size, n_instances, freq_bins, time_bins]
            batch_size, n_instances, freq_bins, time_bins = spectrograms.shape
            # Add channel dimension
            x = spectrograms.view(batch_size * n_instances, 1, freq_bins, time_bins)
        elif len(spectrograms.shape) == 5:
            # If input is [batch_size, n_instances, channels, freq_bins, time_bins]
            batch_size, n_instances, channels, freq_bins, time_bins = spectrograms.shape
            x = spectrograms.view(batch_size * n_instances, channels, freq_bins, time_bins)
        else:
            raise ValueError(f"Expected spectrogram shape of 4 or 5 dimensions, got {len(spectrograms.shape)}")
        
        # Verify spectrogram dimensions
        assert freq_bins == 129 and time_bins == 235, \
            f"Expected spectrogram shape with dims (129, 235), got ({freq_bins}, {time_bins})"
        
        # Process features dimensionality
        if len(features.shape) == 2:
            # If features are [batch_size * n_instances, feature_dim]
            features = features.view(batch_size, n_instances, -1)
        elif len(features.shape) == 3:
            # If features are already [batch_size, n_instances, feature_dim]
            assert features.shape[0] == batch_size and features.shape[1] == n_instances, \
                f"Feature shape mismatch: expected ({batch_size}, {n_instances}, _), got {features.shape}"
        else:
            raise ValueError(f"Expected features shape of 2 or 3 dimensions, got {len(features.shape)}")
        
        # Skip connections processing
        skip_connections = []
        for block in self.spec_encoder:
            x = block(x)
            skip_connections.append(x)
        
        # Extract context-aware features
        spec_features = self.context_pool(x)
        spec_features = spec_features.view(batch_size, n_instances, -1)
        
        # Process temporal features
        temp_features = self.temporal_encoder(features)
        
        # Combine features
        combined = torch.cat([spec_features, temp_features], dim=-1)
        
        # Add positional encoding with dimension checking
        pe = self.pos_encoding.pe[:, :combined.size(1), :combined.size(-1)]
        combined = combined + pe
        
        # Self-attention with masking
        mask = torch.arange(n_instances, device=spectrograms.device).unsqueeze(0) < num_instances.unsqueeze(1)
        attended_features, attention_weights = self.self_attention(
            combined, combined, combined,
            key_padding_mask=~mask
        )
        
        # Score instances with gating
        instance_scores = self.instance_scorer(torch.cat([attended_features, combined], dim=-1))
        instance_scores = instance_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attention_weights = F.softmax(instance_scores / self.temperature, dim=1)
        
        # Weighted pooling
        bag_embedding = torch.sum(attended_features * attention_weights, dim=1)
        
        # Multi-head classification
        logits = []
        for classifier in self.classifier:
            logits.append(classifier(bag_embedding))
        logits = torch.mean(torch.stack(logits, dim=1), dim=1)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'instance_embeddings': attended_features,
            'skip_connections': skip_connections,
            'raw_scores': instance_scores
        }
