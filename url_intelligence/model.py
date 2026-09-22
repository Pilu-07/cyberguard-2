import torch
import torch.nn as nn
import torch.nn.functional as F

class CharURLEncoder(nn.Module):
    """
    Learned Character/Subword URL Representation Branch.
    Uses Character Embeddings + 1D Multi-Scale Convolutions + Global Max Pooling.
    Extracts n-gram subword patterns (e.g. 'secure-login', '.xyz/update', 'login.php').
    """
    def __init__(self, vocab_size=100, embed_dim=48, out_dim=128, kernel_sizes=[3, 5, 7], dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Parallel convolutions to capture 3-gram, 5-gram, and 7-gram URL character patterns
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=out_dim // len(kernel_sizes), kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_dim // len(kernel_sizes)),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        
        total_conv_out = (out_dim // len(kernel_sizes)) * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(total_conv_out, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, char_input):
        # char_input: [batch_size, seq_len]
        embeds = self.embedding(char_input)  # [batch_size, seq_len, embed_dim]
        embeds = embeds.permute(0, 2, 1)    # [batch_size, embed_dim, seq_len]
        
        pooled_outputs = []
        for conv in self.convs:
            c = conv(embeds)               # [batch_size, out_channels, seq_len]
            p = torch.max(c, dim=2)[0]     # Global max pooling: [batch_size, out_channels]
            pooled_outputs.append(p)
            
        cat = torch.cat(pooled_outputs, dim=1)
        return self.proj(cat)              # [batch_size, out_dim]


class EngineeredURLFeatureMLP(nn.Module):
    """
    MLP Projection Branch for the 42 Deterministic URL Features.
    Architecture: Linear(42, 64) -> LayerNorm(64) -> GELU -> Dropout -> Linear(64, 64) -> LayerNorm(64) -> GELU.
    """
    def __init__(self, in_features=42, hidden_dim=64, out_dim=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.mlp(x)


class HybridURLIntelligenceModel(nn.Module):
    """
    Ultra Hi-Fi Hybrid URL Intelligence Classifier.
    Fuses the learned Character/Token branch (128-d) with the 42-d Engineered Cyber Feature branch (64-d).
    Total Fused Vector: 192-d -> 128-d Head -> LayerNorm -> Dropout -> 2 Logits.
    """
    def __init__(self, vocab_size=100, embed_dim=48, char_out_dim=128, 
                 num_engineered_feats=42, mlp_out_dim=64, hidden_dim=128, 
                 num_classes=2, dropout=0.2):
        super().__init__()
        
        self.char_encoder = CharURLEncoder(vocab_size=vocab_size, embed_dim=embed_dim, out_dim=char_out_dim, dropout=dropout)
        self.feature_mlp = EngineeredURLFeatureMLP(in_features=num_engineered_feats, hidden_dim=mlp_out_dim, out_dim=mlp_out_dim, dropout=dropout)
        
        fused_dim = char_out_dim + mlp_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, char_input, feat_input):
        char_rep = self.char_encoder(char_input)    # [B, 128]
        feat_rep = self.feature_mlp(feat_input)     # [B, 64]
        
        fused = torch.cat([char_rep, feat_rep], dim=1) # [B, 192]
        logits = self.classifier(fused)                # [B, 2]
        return logits
