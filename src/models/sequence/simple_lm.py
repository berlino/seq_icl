# Copyright (c) 2023, Tri Dao, Dan Fu.
# Simplified, mostly standalone version of LongConvLM for synthetics.

import math
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from einops import rearrange

from src.utils import instantiate
import src.utils.registry as registry

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, linear_attention=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.linear_attention = linear_attention

    def forward(self, qkv, causal=None, key_padding_mask=None, return_attention=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        mask = None
        if key_padding_mask is not None:
            if self.linear_attention:
                padding_mask = torch.full((batch_size, seqlen), 0.0, dtype=scores.dtype, device=scores.device)
                padding_mask.masked_fill_(key_padding_mask, 1.0)
            else:
                padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                        device=scores.device)
                padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            mask = rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            if self.linear_attention:
                causal_mask = torch.triu(torch.full((seqlen, seqlen), 1.0, device=scores.device, dtype=scores.dtype), 1)
                # take not
                causal_mask = 1.0 - causal_mask
            else:
                causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device, dtype=scores.dtype), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            if mask is None:
                mask = causal_mask
            else:
                if self.linear_attention:
                    mask = mask * causal_mask
                else:
                    mask = mask + causal_mask

        if self.linear_attention:
            if mask is not None:
                scores = scores * mask.to(scores.dtype)
            attention = scores
        else:
            if mask is not None:
                scores = scores + mask.to(scores.dtype)
            attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)

        if return_attention:
            return output, attention
        else:
            return output

class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 linear_attention=False,
                 softmax_scale=None, causal=False, layer_idx=None, dwconv=False,return_residual=False,device=None, dtype=None) -> None:
        """
            return_residual: whether to return the input x along with the output. This is for
                performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual
        self.linear_attention = linear_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisibl e by num_heads"
        self.head_dim = self.embed_dim // num_heads

        linear_cls = nn.Linear
        linear_resid_cls = LinearResidual
        inner_attn_cls =  SelfAttention

        if not self.return_residual:
            self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                        groups=3 * embed_dim)

        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, linear_attention=linear_attention,
                                         attention_dropout=dropout)

        # if self.linear_attention:
        #     self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False).to(device=device, dtype=dtype)

        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, return_attention=False, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """

        kwargs = ({'key_padding_mask': key_padding_mask, **kwargs})

        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                            'b d s -> b s d').contiguous()
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        context = self.inner_attn(qkv, return_attention=return_attention, **kwargs)

        if return_attention:
            context, attentions = context
            if self.return_residual:
                x = (x, (context, attentions))
            else:
                x = ((context, attentions), )

        # context = self.group_norm(context)

        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        if self.return_residual or return_attention:
            return (out, x)
        else:
            return out

class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None,
                 word_embed_proj_dim=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                                **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False,
                                        **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0.,
                 return_residual=False,
                 return_attention=False,
                 residual_in_fp32=False):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        self.return_attention = return_attention
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        attentions = None
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, return_attention=self.return_attention, **mixer_kwargs)
            if self.return_attention:  # mixer out is actually a pair here
                hidden_states, attentions = hidden_states
                attentions = attentions[0]
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)

            return hidden_states, residual, attentions

        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            if self.return_attention:
                hidden_states, attentions = hidden_states

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))
            if self.return_attention:
                return hidden_states, attentions
            else:
                return hidden_states

def create_mixer_cls(layer=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None, linear_attention=False,
                     device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)

        mha_cls = MHA

        mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx, linear_attention=linear_attention,
                            **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    else:
        mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **(attn_cfg if attn_cfg is not None else {}), **factory_kwargs)
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls


def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None, linear_attention=False, return_attention=False,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer,
                                 linear_attention=linear_attention,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx,
                                 **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32, return_attention=return_attention)
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class LMBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False, return_attention=False,
                 device=None, dtype=None, linear_attention=False, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
                                             **factory_kwargs)


        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            linear_attention=linear_attention,
            return_attention=return_attention,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None, return_hidden_outputs=False):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids,)
        residual = None
        if return_hidden_outputs:
            hidden_outputs = [hidden_states.detach().cpu()]
            attention_outputs = []
        else:
            hidden_outputs = None
            attention_outputs = None

        for layer in self.layers:
            hidden_states, residual, attentions = layer(hidden_states, residual)
            if return_hidden_outputs:
                hidden_outputs.append(hidden_states.detach().cpu())
                if attention_outputs is not None and attentions is not None:
                    attention_outputs.append((attentions[0].detach().cpu(), attentions[1].detach().cpu()))

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        hidden_states = {"hidden_states": hidden_states, "hidden_outputs": hidden_outputs, "attention_outputs": attention_outputs}

        return hidden_states


class SimpleLMHeadModel(nn.Module):
    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1,
                 linear_attention=False,
                 return_attention=False,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            linear_attention=linear_attention,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            return_attention=return_attention,
            **factory_kwargs, **kwargs
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, state=None, return_hidden_outputs=False): # state for the repo interface
        hidden_states = self.backbone(input_ids, position_ids=position_ids, return_hidden_outputs=return_hidden_outputs)
        hidden_states, hidden_outputs = hidden_states["hidden_states"], (hidden_states["hidden_outputs"], hidden_states["attention_outputs"])
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits), hidden_outputs, None


class SimpleLMHeadModelNoFFN(nn.Module):
    """
    Same as HeadModel but without the MLP in the Transformer block, suitable for LSTM.
    """

    def __init__(self, d_model: int, vocab_size: int,
                 layer=None, max_position_embeddings=-1,
                 n_layer=1,
                 embed_dropout: float = 0.1,
                 pad_vocab_size_multiple: int = 1,
                 device=None, dtype=None, **kwargs) -> None:
        """
        Args:
            max_position_embeddings: if <= 0, no position embeddings
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,                                              **factory_kwargs)
        self.embed_dropout = nn.Dropout(embed_dropout)

        mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=None, **factory_kwargs)

        self.mixer = nn.ModuleList([mixer_cls(d_model) for _ in range(n_layer)])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, state=None,  return_hidden_outputs=False): # state for the repo interface
        embeddings = self.embeddings(input_ids, position_ids=position_ids)
        hidden_states = self.embed_dropout(embeddings)
        if return_hidden_outputs:
            hidden_outputs = [hidden_states.detach().cpu()]
        else:
            hidden_outputs = None

        for mixer in self.mixer:
            hidden_states = mixer(embeddings)
            if return_hidden_outputs:
                hidden_outputs.append(hidden_states.detach().cpu())
        # hidden_states, hidden_outputs = self.mixer(embeddings)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits), hidden_outputs, None


class HybridLMHeadModel(nn.Module):
    """
    Flattern the backbone and the head into a single model.
    """
    def __init__(self, d_model, vocab_size, layers, device=None, dtype=None, resid_dropout=0.0, embed_dropout=0.1, **kwargs):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # does not use positional embeddings
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings=0, **factory_kwargs)

        nn_layers = []
        for layer_name in layers:
            layer_config = layers[layer_name]
            layer_idx = int(layer_name[5:]) # layer0
            mixer_layer_config = layer_config["mixer_layer"]
            d_inner = layer_config["d_inner"]
            nn_layer = create_block(d_model, d_inner=d_inner, layer=mixer_layer_config,  layer_idx=layer_idx, resid_dropout1=embed_dropout if layer_idx == 0 else resid_dropout, resid_dropout2=resid_dropout, **factory_kwargs)
            nn_layers.append(nn_layer)
        self.layers = nn.ModuleList(nn_layers)

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        n_layer = len(self.layers)
        self.apply(partial(_init_weights, n_layer=n_layer))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, state=None, return_hidden_outputs=False):
        assert not return_hidden_outputs # to be supported
        hidden_states = self.embeddings(input_ids, position_ids=position_ids,)
        residual = None

        for layer in self.layers:
            # hidden_states, residual, attentions = layer(hidden_states, residual)
            hidden_states, residual, attentions = layer(hidden_states, residual, mixer_kwargs={"input_ids": input_ids})

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits), None, None