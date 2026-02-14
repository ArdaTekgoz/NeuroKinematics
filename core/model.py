import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility Blocks
# ============================================================

class ResidualBlock(nn.Module):
    """
    Fully connected residual block with optional LayerNorm.
    """

    def __init__(self, dim, dropout=0.0, use_layernorm=True):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        if self.use_layernorm:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        if self.use_layernorm:
            out = self.norm2(out)

        out = out + identity
        out = self.activation(out)

        return out


# ============================================================
# IK Network
# ============================================================

class IKNet(nn.Module):
    """
    Inverse Kinematics Network

    Input:
        pose (B, 7)
        [x, y, z, qx, qy, qz, qw]

    Output:
        joint angles (B, 6)
    """

    def __init__(
        self,
        input_dim=7,
        output_dim=6,
        hidden_dim=256,
        num_res_blocks=3,
        dropout=0.1,
        use_layernorm=True,
        predict_confidence=False
    ):
        super().__init__()

        self.predict_confidence = predict_confidence

        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim,
                dropout=dropout,
                use_layernorm=use_layernorm
            )
            for _ in range(num_res_blocks)
        ])

        # Joint regression head
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Optional confidence head
        if predict_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        self.activation = nn.GELU()

        self._initialize_weights()

    # --------------------------------------------------------
    # Weight Initialization
    # --------------------------------------------------------

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, x):
        """
        x: (B, 7)
        """

        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError(
                f"Expected input shape (B,7), got {x.shape}"
            )

        out = self.input_layer(x)
        out = self.activation(out)

        for block in self.res_blocks:
            out = block(out)

        joints = self.output_layer(out)

        if self.predict_confidence:
            confidence = self.confidence_head(out)
            return joints, confidence

        return joints

    # --------------------------------------------------------
    # Debug Utility
    # --------------------------------------------------------

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print("==== IKNet Summary ====")
        print(f"Trainable Parameters: {self.count_parameters():,}")
        print(f"Predict Confidence: {self.predict_confidence}")
        print("=======================")
