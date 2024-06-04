import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None
from torch import nn
from torch.nn import functional as F


@dataclass
class SelfGenomeGPTConfig:
    conv_size: int = 40
    d_model: int = 256
    n_heads: int = 4
    intermediate_size: int = 1024
    n_layers: int = 4
    norm_eps: float = 1e-12
    max_len: int = 1000
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    attn_impl: str = "flash"  # 'sdpa' or 'flash' or 'slow'

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.initializer_range > 0.0, "initializer_range must be positive"
        assert (
            0.0 <= self.attention_dropout < 1.0
        ), "attention_dropout must be in [0, 1)"
        assert 0.0 <= self.mlp_dropout < 1.0, "mlp_dropout must be in [0, 1)"
        assert self.norm_eps > 0.0, "norm_eps must be positive"
        assert self.conv_size % 2 == 0, "conv_size must be divisible by 2"

    @classmethod
    def from_pretrained(cls, model_path: str):
        """This method is used to load a pretrained model configuration from a given path.

        Args:
            model_path (str): The path to the directory containing the pretrained model's configuration file.

        Returns:
            KiruaConfig: An instance of KiruaConfig initialized with the values from the pretrained model's configuration.
        """
        model_dir = Path(model_path)
        with open(model_dir / "config.json", "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@torch.no_grad()
def init_weights_impl(module, initializer_range: float):
    """Initialize the weights"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class SelfGenomeGPTFFN(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        intermediate_size = config.intermediate_size
        dropout = config.mlp_dropout
        self.lin1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.lin2 = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.silu(self.lin1(x))))


class SelfGenomeGPTCausalSelfAttention(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        if config.attn_impl == "flash":
            assert (
                flash_attn_qkvpacked_func is not None
            ), "flash_attn must be installed if using flash_attn"

        self.config = config
        d_model = config.d_model
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.Wout = nn.Linear(d_model, d_model, bias=False)

        if config.attn_impl == "flash":
            self.qkv_forwar_fn = self._flash_attn_forward
        elif config.attn_impl == "sdpa":
            self.qkv_forwar_fn = self._sdpa_forward
        else:
            raise ValueError(f"Unknown attention implementation: {config.attn_impl}")

    def _sdpa_forward(self, qkv_packed: torch.Tensor):
        """
        qkv_packed: shape (batch_size, seq_len, 3 * d_model)
        """
        qkv = rearrange(
            qkv_packed,
            "... l (three h d) -> ... three h l d",
            three=3,
            h=self.config.n_heads,
        )  # (batch_size, 3, n_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(dim=1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=True,
        )  # (batch_size, n_heads, seq_len, head_dim)
        return rearrange(out, "... h l d -> ... l (h d)")

    def _flash_attn_forward(self, qkv_packed: torch.Tensor):
        qkv = rearrange(
            qkv_packed,
            "... (three h d) -> ... three h d",
            three=3,
            h=self.config.n_heads,
        )
        attn_out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            causal=True,
        )
        return rearrange(attn_out, "... h d -> ... (h d)")

    def forward(self, x):
        qkv = self.Wqkv(x)
        attn_out = self.qkv_forwar_fn(qkv)
        out = self.Wout(attn_out)
        return out


class SelfGenomeGPTLayer(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = SelfGenomeGPTCausalSelfAttention(config)
        self.ffn = SelfGenomeGPTFFN(config)

    def forward(self, x):
        hid = self.attn(self.norm1(x)) + x
        out = self.ffn(self.norm2(hid)) + hid
        return out


class SelfGenomeGPTMLPHead(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SelfGenomeGPTPositionalEncoding(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.stride = config.conv_size // 2
        max_len = config.max_len // self.stride

        self.register_buffer(
            "position_ids",
            torch.arange(max_len).expand((1, -1)),
            persistent=False,
        )
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        x += position_embeddings
        return x


@dataclass
class SelfGenomeGPTOutput:
    hidden_states: torch.Tensor
    head_logits: torch.Tensor


class SelfGenomeGPT(nn.Module):
    def __init__(self, config: SelfGenomeGPTConfig):
        super().__init__()
        self.config = config
        self.stride = config.conv_size // 2
        self.in_conv = nn.Conv1d(
            in_channels=4,
            out_channels=config.d_model,
            kernel_size=config.conv_size,
            stride=self.stride,
            padding=self.stride,
        )
        self.pos_encoding = SelfGenomeGPTPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [SelfGenomeGPTLayer(config) for _ in range(config.n_layers)]
        )
        self.out_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.head = SelfGenomeGPTMLPHead(config)

    def forward(self, x: torch.LongTensor) -> SelfGenomeGPTOutput:
        """SelfGenomeGPT forward pass

        Args:
            x (torch.LongTensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            SelfGenomeGPTOutput: Output of the model, including hidden states and head logits
        """
        bs, seqlen = x.shape
        seqlen_for_transformer = seqlen // self.stride

        x_onehot = F.one_hot(x, num_classes=4)
        x_input = rearrange(x_onehot, "b l c -> b c l").float()  # for conv1d
        x_input = self.in_conv(x_input)[:, :, 1:]
        x_input = rearrange(
            x_input, "b c l -> b l c"
        )  # back to (batch_size, seq_len, d_model)

        x_input = self.pos_encoding(x_input)

        for layer in self.layers:
            x_input = layer(x_input)
        x_input = self.out_norm(x_input)

        out = SelfGenomeGPTOutput(hidden_states=x_input, head_logits=self.head(x_input))
        return out

    def save_pretrained(self, save_directory: str):
        """Save a model and its configuration file to a directory, so that it can be re-loaded using the `from_pretrained` class method.

        Args:
            save_directory (str): Directory to which to save the model.
        """
        model_dir = Path(save_directory)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        torch.save(self.state_dict(), model_dir / "model.pth")

    @classmethod
    def from_pretrained(cls, model_path: str):
        """This method is used to load a pretrained model from a given path.

        Args:
            model_path (str): The path to the directory containing the pretrained model's configuration file.

        Returns:
            Kirua: An instance of Kirua initialized with the weights from the pretrained model.
        """
        model_dir = Path(model_path)
        config = SelfGenomeGPTConfig.from_pretrained(model_path)
        model = cls(config)
        model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
        return model

    @classmethod
    def from_config(cls, config: SelfGenomeGPTConfig):
        """This method is used to load a pretrained model from a given path.

        Args:
            model_path (str): The path to the directory containing the pretrained model's configuration file.

        Returns:
            Kirua: An instance of Kirua initialized with the weights from the pretrained model.
        """
        model = cls(config)
        return model
