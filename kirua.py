import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import torch
from einops import rearrange
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import GatedMlp
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
from flash_attn.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
from torch import nn
from torch.nn import functional as F

from ..constants import *


@dataclass
class KiruaConfig:
    d_model: int = 256
    in_model: int = 1280
    n_heads: int = 8
    intermediate_size: int = 2048
    n_layers: int = 6
    norm_eps: float = 1e-12
    expr_bin: int = 1000
    initializer_range: float = 0.02
    attention_out_bias: bool = True
    norm_type: str = "layer_norm"  # layer_norm or rms_norm
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    mlp_in_proj: bool = False

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
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def apply_norm(
    norm_module: nn.Module, x: torch.Tensor, is_layer_norm: bool = True
) -> torch.Tensor:
    return norm_module(x)  # torch built native ops
    # return layer_norm_fn(
    #     x, weight=norm_module.weight, bias=norm_module.bias, eps=norm_module.eps
    # ) if is_layer_norm else norm_module(x)


class KiruaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # def _norm(self, x):
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     output = self._norm(x.float()).type_as(x)
    #     return output * self.weight

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # new implementation from OLMo: https://github.com/allenai/OLMo/blob/2eae98887629f8ed20a61049facce9820170a4d6/olmo/model.py#L222
    #     with torch.autocast(enabled=False, device_type=x.device.type):
    #         og_dtype = x.dtype
    #         x = x.to(torch.float32)
    #         variance = x.pow(2).mean(-1, keepdim=True)
    #         x = x * torch.rsqrt(variance + self.eps)
    #         x = x.to(og_dtype)

    #     return self.weight * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_fn(x, weight=self.weight, bias=None, eps=self.eps)


class KiruaLayer(nn.Module):
    def __init__(self, config: KiruaConfig):
        """Initializes a new instance of the KiruaLayer class.

        Args:
            config (KiruaConfig): An instance of KiruaConfig class which contains the configuration parameters for the KiruaLayer.
        """
        super().__init__()

        self.norm_type = config.norm_type
        self.is_layer_norm = self.norm_type == "layer_norm"
        if self.norm_type == "layer_norm":
            NORM_MODULE = nn.LayerNorm
        elif self.norm_type == "rms_norm":
            NORM_MODULE = KiruaRMSNorm
        else:
            raise ValueError(f"Unsupported norm type: {config.norm_type}")
        self.norm1 = NORM_MODULE(config.d_model, config.norm_eps)
        self.norm2 = NORM_MODULE(config.d_model, config.norm_eps)

        self.mlp = GatedMlp(
            in_features=config.d_model,
            hidden_features=config.intermediate_size,
            out_features=config.d_model,
            activation=F.silu,
            bias1=False,
            bias2=False,
        )
        self.mlp_dropout = config.mlp_dropout
        self.mha = MHA(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            qkv_proj_bias=False,
            out_proj_bias=config.attention_out_bias,
            dropout=config.attention_dropout,
            use_flash_attn=True,
        )

    def forward(self, x: torch.Tensor, **mha_kwargs):
        """Forward pass for the KiruaLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (total_token_count, hidden_size).
            **mha_kwargs (dict): Additional keyword arguments for the multi-head attention (MHA) layer.
                                Must include 'cu_seqlens' and 'max_seqlen' if using FlashAttention.
                                'cu_seqlens' is a tensor of shape (batch_size + 1,) and dtype torch.int32,
                                representing the cumulative sequence lengths of the sequences in the batch.
                                'max_seqlen' is an integer representing the maximum sequence length in the batch.

        Returns:
            torch.Tensor: Output tensor after passing through the layer.
        """
        x = x + self.mha(
            apply_norm(self.norm1, x, is_layer_norm=self.is_layer_norm), **mha_kwargs
        )
        x = x + (
            F.dropout(
                self.mlp(apply_norm(self.norm2, x, is_layer_norm=self.is_layer_norm)),
                p=self.mlp_dropout,
                training=self.training,
            )
        )
        return x

    def _self_attn(self, qkv: torch.Tensor):
        """mha self attn with only one batch
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (S, 3, H, D)

        Returns
        -------
            output: The tensor containing the output of the self-attention layer. (S, H, D)
            attention: The tensor containing the attention weights. (H, S, S)
        """
        q, k, v = qkv.unbind(dim=1)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("thd,shd->hts", q, k * softmax_scale)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.mha.inner_attn.drop(attention)
        output = torch.einsum("hts,shd->thd", attention_drop, v)
        return output, attention

    def _mha_slow_forward(
        self,
        x,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """
        this is the slow version of mha forward, the only use case is to get the attention weights
        so the batch size can only be 1

        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.

            (the following args are not used, only for batch size assertion to make sure it's 1)
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
        """
        assert (
            cu_seqlens.shape[0] == 2
        ), f"batch size must be 1, but got {cu_seqlens.shape[0]}"
        assert (
            max_seqlen == x.shape[0]
        ), f"max_seqlen must be equal to seqlen, but got {max_seqlen} != {x.shape[0]}"
        mha_module = self.mha

        qkv = mha_module.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=mha_module.head_dim
        )  # (seqlen of the only one sample in batch, 3, nHead, head Dim)
        context, attn_weights = self._self_attn(qkv)
        out = mha_module.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return (out, attn_weights)

    def _mlp_slow_forward(self, x: torch.Tensor):
        mlp = self.mlp
        if x.device.type != "cpu":
            # fast path
            return mlp(x)

        # cpu can only use native ops
        y = mlp.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * mlp.activation(gate)
        y = mlp.fc2(y)
        return y

    def layer_slow_forward(self, x: torch.Tensor, **mha_kwargs):
        """Slow forward pass for the KiruaLayer just for visualization of attention weights

        Args:
            x (torch.Tensor): Input tensor of shape (total_token_count, hidden_size).
            **mha_kwargs (dict): Additional keyword arguments for the multi-head attention (MHA) layer.
                                Must include 'cu_seqlens' and 'max_seqlen' if using FlashAttention.
                                'cu_seqlens' is a tensor of shape (batch_size + 1,) and dtype torch.int32,
                                representing the cumulative sequence lengths of the sequences in the batch.
                                'max_seqlen' is an integer representing the maximum sequence length in the batch.

        Returns:
            torch.Tensor: Output tensor after passing through the layer.
            torch.Tensor: Attention weights tensor.
        """
        attn_out, attn_weight = self._mha_slow_forward(
            apply_norm(self.norm1, x, is_layer_norm=self.is_layer_norm),
            **mha_kwargs,
        )
        hid = x + attn_out
        out = hid + (
            F.dropout(
                self._mlp_slow_forward(
                    apply_norm(self.norm2, hid, is_layer_norm=self.is_layer_norm)
                ),
                p=self.mlp_dropout,
                training=self.training,
            )
        )
        return out, attn_weight


class KiruaPooler(nn.Module):
    def __init__(self, config: KiruaConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ):
        """
        Args:
            hidden_states: (total_token_count, hidden_size)
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, can be used to get the cls token of concat sequences.
                not used if None
        """
        first_token_tensor = (
            hidden_states[:, 0]
            if cu_seqlens is None
            else hidden_states[cu_seqlens[:-1]]
        )
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class KiruaEmbedding(nn.Module):
    def __init__(self, config: KiruaConfig, emb_file: Path):
        super().__init__()
        # special tokens
        self.gene_num_special_tokens = GENE_NUM_SPECIAL_TOKENS
        self.gene_pad_idx = GENE_PAD_IDX

        self.expr_pad_idx = EXPR_PAD_IDX
        self.expr_cls_idx = EXPR_CLS_IDX
        self.expr_mask_idx = EXPR_MASK_IDX
        self.expr_num_special_tokens = EXPR_NUM_SPECIAL_TOKENS

        self.config = config

        # freeze gene embeddings
        protein_emb: torch.Tensor = torch.load(emb_file)
        assert (
            protein_emb.shape[1] == config.in_model
        ), f"Dimension mismatch protein_emb.shape[1] {protein_emb.shape[1]} != config.in_model {config.in_model}"
        dummy_emb = torch.zeros(
            self.gene_num_special_tokens, protein_emb.shape[1], dtype=protein_emb.dtype
        )
        protein_emb = torch.cat(
            [
                dummy_emb,
                protein_emb.to(dummy_emb.device),
            ],
            dim=0,
        )  # concat dummy emb for special tokens
        self.register_buffer(
            "protein_emb", protein_emb, persistent=False
        )  # not saved in state_dict, too large
        self.protein_emb_dim = config.in_model

        # expr level binning emb
        self.expr_emb = nn.Embedding(
            config.expr_bin + self.expr_num_special_tokens,
            config.d_model,
            padding_idx=self.expr_pad_idx,
        )

    @property
    def gene_emb_size(self):
        return self.protein_emb.shape

    @property
    def expr_emb_size(self):
        return self.expr_emb.weight.shape

    def gene(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return F.embedding(input_ids, self.protein_emb)

    def expr(self, expr_bins: torch.LongTensor) -> torch.Tensor:
        return self.expr_emb(expr_bins)


@dataclass
class KiruaOutput:
    sequence_output: torch.Tensor
    pooler_output: torch.Tensor = None
    attention_weights: list[torch.Tensor] = None


class Kirua(nn.Module):
    def __init__(
        self, config: KiruaConfig, emb_file: Path, add_pooling_layer: bool = True
    ):
        super().__init__()
        self.config = config

        # gene + expression level embedding module
        self.embeddings = KiruaEmbedding(config, emb_file)

        # gene emb space to model emb space
        self.in_proj = (
            nn.Linear(config.in_model, config.d_model)
            if not config.mlp_in_proj
            else nn.Sequential(
                nn.Linear(config.in_model, config.in_model),
                nn.SiLU(),
                nn.Linear(config.in_model, config.d_model),
            )
        )

        # transformer layers
        self.blocks = nn.ModuleList(
            [KiruaLayer(config) for _ in range(config.n_layers)]
        )

        # output norm (after last layer)
        self.norm_type = config.norm_type
        self.is_layer_norm = self.norm_type == "layer_norm"
        if self.norm_type == "layer_norm":
            NORM_MODULE = nn.LayerNorm
        elif self.norm_type == "rms_norm":
            NORM_MODULE = KiruaRMSNorm
        else:
            raise ValueError(f"Unsupported norm type: {config.norm_type}")
        self.out_norm = NORM_MODULE(config.d_model, config.norm_eps)

        # pooler (optional)
        self.pooler = KiruaPooler(config) if add_pooling_layer else None

    def forward(self, input_ids: torch.Tensor, expr_bins: torch.Tensor, **mha_kwargs):
        """
        input_ids: (total_token_count, ), dtype torch.int64. The input gene token ids.
        expr_bins: (total_token_count, ), dtype torch.int64. The input expression level bin ids.
        mha_kwargs: dict
            need to have
                cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                    of the sequences in the batch, used to index into x. Only applicable when using
                    FlashAttention.
                max_seqlen: int. Maximum sequence length in the batch.
        """
        hid = self.embeddings.gene(input_ids)
        hid = self.in_proj(hid)
        hid = hid + self.embeddings.expr(expr_bins)

        for block in self.blocks:
            hid = block(hid, **mha_kwargs)
        hid = apply_norm(self.out_norm, hid, is_layer_norm=self.is_layer_norm)
        out = KiruaOutput(
            sequence_output=hid,
            pooler_output=(
                self.pooler(hid, mha_kwargs.get("cu_seqlens", None))
                if self.pooler is not None
                else None
            ),
        )
        return out

    def model_slow_forward(
        self, input_ids: torch.Tensor, expr_bins: torch.Tensor, **mha_kwargs
    ):
        """
        input_ids: (total_token_count, ), dtype torch.int64. The input gene token ids.
        expr_bins: (total_token_count, ), dtype torch.int64. The input expression level bin ids.
        mha_kwargs: dict
            need to have
                cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                    of the sequences in the batch, used to index into x. Only applicable when using
                    FlashAttention.
                max_seqlen: int. Maximum sequence length in the batch.
        """
        hid = self.embeddings.gene(input_ids)
        hid = self.in_proj(hid)
        hid = hid + self.embeddings.expr(expr_bins)

        attn_weights = []
        block: KiruaLayer
        for block in self.blocks:
            rets = block.layer_slow_forward(hid, **mha_kwargs)
            hid = rets[0]
            attn_weights.append(rets[1])
        hid = apply_norm(self.out_norm, hid, is_layer_norm=self.is_layer_norm)
        out = KiruaOutput(
            sequence_output=hid,
            pooler_output=(
                self.pooler(hid, mha_kwargs.get("cu_seqlens", None))
                if self.pooler is not None
                else None
            ),
            attention_weights=attn_weights,
        )
        return out

    def save_pretrained(self, save_directory: str):
        """
        Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `from_pretrained` class method.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=1)

        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        emb_file: Path,
        config: KiruaConfig | None = None,
        *inputs,
        **kwargs,
    ):
        """
        Instantiate a Kirua model from a pre-trained model file.
        """
        model_dir = Path(model_path)
        if config is None:
            with open(model_dir / "config.json", "r") as f:
                config_dict = json.load(f)
            config = KiruaConfig(**config_dict)
        model = cls(config, emb_file, *inputs, **kwargs)
        state_dict = torch.load(model_dir / "pytorch_model.bin")
        model.load_state_dict(state_dict, strict=False)
        return model


@dataclass
class KiruaPreTrainingOutput:
    loss: torch.Tensor = None
    mlm_loss: torch.Tensor = None
    sequence_output: torch.Tensor = None
    pred_expr_embs: torch.Tensor = None


class LossType(str, Enum):
    CE = "cross_entropy"
    MSE = "mse"
    MAE = "mae"


class KiruaForPretraining(nn.Module):
    def __init__(
        self,
        config: KiruaConfig,
        emb_file: Path,
        loss_type: LossType = LossType.CE,
        transform_before_head: bool = False,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.loss_type = loss_type
        self.transform_before_head = transform_before_head

        # kirua model
        self.model = Kirua(config, emb_file, add_pooling_layer=False)

        if self.transform_before_head:
            self.trans_before_head = nn.Linear(config.d_model, config.d_model)

        if loss_type == LossType.CE:
            # project model output hidden state to expr bin size
            self.expr_head = nn.Linear(
                config.d_model, self.model.embeddings.expr_emb_size[0], bias=False
            )
            self.loss_fn = cross_entropy_loss
            if tie_weights:
                self.expr_head.weight = self.model.embeddings.expr_emb.weight

        elif loss_type in (LossType.MSE, LossType.MAE):
            self.expr_head = nn.Linear(config.d_model, 1)
            if loss_type == LossType.MSE:
                self.loss_fn = nn.MSELoss()
            else:
                self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def init_weights(self):
        # init weights
        self.apply(
            partial(init_weights_impl, initializer_range=self.config.initializer_range)
        )

    def calc_loss(
        self,
        pred_expr_embs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == LossType.CE:
            loss = self.loss_fn(pred_expr_embs, labels)[0].mean()
        else:
            loss = self.loss_fn(pred_expr_embs.view(-1), labels.view(-1))
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        expr_bins: torch.LongTensor,
        masked_tokens_mask: torch.Tensor,
        labels: torch.LongTensor,
        **mha_kwargs,
    ) -> KiruaPreTrainingOutput:
        kirua_output: KiruaOutput = self.model(input_ids, expr_bins, **mha_kwargs)

        emb = kirua_output.sequence_output[masked_tokens_mask]
        if self.transform_before_head:
            emb = self.trans_before_head(emb)
        pred_expr_embs = self.expr_head(emb)
        mlm_loss = self.calc_loss(pred_expr_embs, labels)
        return KiruaPreTrainingOutput(
            loss=mlm_loss,
            mlm_loss=mlm_loss.detach().clone(),
            sequence_output=kirua_output.sequence_output,
            pred_expr_embs=pred_expr_embs,
        )

    def noisy_forward(
        self,
        input_ids: torch.LongTensor,
        expr_bins: torch.LongTensor,
        masked_tokens_mask: torch.Tensor,
        labels: torch.LongTensor,
        noise_std: float,
        **mha_kwargs,
    ) -> KiruaPreTrainingOutput:
        """
        Modified base model forward that add gaussian noise on protein emb, to make model explore more protein space
        """
        gene_embs = self.model.embeddings.gene(input_ids)

        ## TODO: to figure out how to add noise to protein emb is better
        gene_embs += torch.randn_like(gene_embs) * noise_std

        hid = self.model.in_proj(gene_embs)
        hid = hid + self.model.embeddings.expr(expr_bins)

        for block in self.model.blocks:
            hid = block(hid, **mha_kwargs)
        hid = apply_norm(
            self.model.out_norm, hid, is_layer_norm=self.model.is_layer_norm
        )

        emb = hid[masked_tokens_mask]
        if self.transform_before_head:
            emb = self.trans_before_head(emb)
        pred_expr_embs = self.expr_head(emb)
        mlm_loss = self.calc_loss(pred_expr_embs, labels)
        return KiruaPreTrainingOutput(
            loss=mlm_loss,
            mlm_loss=mlm_loss.detach().clone(),
            sequence_output=hid,
            pred_expr_embs=pred_expr_embs,
        )


def get_cell_embeddings(
    sequence_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Get cell-level embeddings from the gene-level sequence output of Kirua model.

    Args:
        sequence_output (torch.Tensor): (total_nnz, d_model) The output of Kirua model.
        cu_seqlens (torch.Tensor): (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        method (str, optional): choice: ['mean', 'cls'], default 'mean'
            mean: mean pooling of all gene-level embeddings
            cls: use the cls token of each sequence
                 (not recommended for pretrained model, currently cls token is not trained during pretraining)

    Returns:
        torch.Tensor: _description_
    """
    assert (
        cu_seqlens[-1] == sequence_output.shape[0]
    ), f"cu_seqlens total != sequence_output total, {cu_seqlens[-1]} != {sequence_output.shape[0]}"

    if method == "mean":
        cu_seqlens_list = cu_seqlens.tolist()
        embs = []
        for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
            embs.append(
                # skip cls token
                sequence_output[start + 1 : end].mean(dim=0)
            )
        embs = torch.stack(embs)
    elif method == "cls":
        embs = sequence_output[cu_seqlens[:-1]]
    else:
        raise ValueError(f"Unsupported method: {method}")
    return embs


if __name__ == "__main__":
    config = KiruaConfig(1000)
    device = "cuda:0"
    dt = torch.bfloat16
    model = Kirua(config).to(device=device, dtype=dt)

    inputs = torch.randn(100, 1280, dtype=dt, device=device)
    cu_seqlens = torch.tensor([0, 50, 100], dtype=torch.int32, device=device)

    out: KiruaOutput = model(inputs, cu_seqlens=cu_seqlens, max_seqlen=50)

    print(out)
    print(out.sequence_output.shape)
    print(out.pooler_output.shape)

    model.save_pretrained("weights/kirua")

    new_model = Kirua.from_pretrained("weights/kirua", add_pooling_layer=False)

    print(model)
    print(new_model)
