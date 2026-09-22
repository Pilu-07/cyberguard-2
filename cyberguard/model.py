import os
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CyberFeatureMLP(nn.Module):
    """
    MLP projection head for the 21-dimensional deterministic cyber features.
    Architecture: Linear(21, 64) -> LayerNorm(64) -> GELU -> Dropout -> Linear(64, 64) -> LayerNorm(64) -> GELU.
    """
    def __init__(self, in_features=21, hidden_dim=64, out_dim=64, dropout=0.2):
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

class CyberGuardHybridModel(nn.Module):
    """
    CYBERGUARD Hybrid Model Architecture.
    Combines 768-d semantic text branch with 21-d programmatic cyber feature branch (projected to 64-d).
    Fused dimension: 768 + 64 = 832-d -> 256-d Hidden Head -> 2-class Logits.
    """
    def __init__(self, model_name="microsoft/deberta-v3-base", text_dim=768, cyber_in_dim=21, 
                 cyber_out_dim=64, hidden_dim=256, num_classes=2, dropout=0.2):
        super().__init__()
        
        self.model_name = model_name
        self.text_dim = text_dim
        self.cyber_in_dim = cyber_in_dim
        self.cyber_out_dim = cyber_out_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Transformer backbone if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.config = AutoConfig.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
                self.text_hidden_size = self.config.hidden_size
            except Exception:
                self.encoder = None
                self.text_hidden_size = text_dim
        else:
            self.encoder = None
            self.text_hidden_size = text_dim
            
        # Branch B: Cyber Feature Engine MLP
        self.cyber_mlp = CyberFeatureMLP(
            in_features=cyber_in_dim, 
            hidden_dim=cyber_out_dim, 
            out_dim=cyber_out_dim, 
            dropout=dropout
        )
        
        # Feature Fusion (768 + 64 = 832)
        fused_dim = self.text_hidden_size + cyber_out_dim
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, cyber_features=None, text_embeddings=None):
        """
        Forward pass. Accepts either raw tokens (input_ids) or dense text_embeddings (768-d).
        """
        if text_embeddings is not None:
            text_rep = text_embeddings
        elif self.encoder is not None and input_ids is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_rep = outputs.pooler_output
            else:
                text_rep = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Either text_embeddings (768-d) or input_ids must be provided.")
            
        if cyber_features is None:
            raise ValueError("cyber_features tensor must be provided for hybrid model forward pass.")
            
        cyber_rep = self.cyber_mlp(cyber_features)
        
        # Feature Fusion
        fused_rep = torch.cat([text_rep, cyber_rep], dim=1)
        
        # Classification Logits
        logits = self.classifier(fused_rep)
        return logits

    def save_weights(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path, device=torch.device('cpu')):
        self.load_state_dict(torch.load(path, map_location=device))

class CyberGuardSemanticBaselineModel(nn.Module):
    """
    Stage 2 Baseline: Semantic Branch Only (Text + Channel representation).
    Direct 768-d -> 256-d Hidden Head -> 2-class Logits without cyber signals.
    """
    def __init__(self, text_dim=768, hidden_dim=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_embeddings):
        return self.classifier(text_embeddings)

if __name__ == '__main__':
    print("Testing CyberGuardHybridModel Architecture...")
    model = CyberGuardHybridModel()
    dummy_text_emb = torch.randn(4, 768)
    dummy_cyber_feat = torch.randn(4, 21)
    out = model(text_embeddings=dummy_text_emb, cyber_features=dummy_cyber_feat)
    print("Hybrid Logits Shape:", out.shape)
    
    print("Testing CyberGuardSemanticBaselineModel Architecture...")
    baseline_model = CyberGuardSemanticBaselineModel()
    b_out = baseline_model(dummy_text_emb)
    print("Baseline Logits Shape:", b_out.shape)
