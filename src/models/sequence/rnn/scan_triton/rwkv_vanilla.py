# copied from https://github.com/codekansas/rwkv/blob/master/rwkv/triton/wkv/vanilla.py
from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


AUTOTUNE_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK_SIZE_C": 32}, num_warps=2),
    triton.Config({"BLOCK_SIZE_C": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=8),
]


def get_block_size_c(chans: int) -> int:
    if chans < 32:
        return 32
    if chans < 64:
        return 64
    return 128

# @triton.autotune(configs=AUTOTUNE_CONFIGS, key=["chans"])
@triton.jit
def wkv_triton_vanilla_forward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_ab,
    state_s_c,
    # WKV
    wkv_ptr,
    wkv_s_b,
    wkv_s_t,
    wkv_s_c,
    # Output state
    state_out_ptr,
    state_out_s_b,
    state_out_s_ab,
    state_out_s_t,
    state_out_s_c,
    # Params
    chans,
    tsz,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    cs = (c_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_ab

    # Pointers to the batch (and possibly channel) for the output tensors.
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_ab

    # Loads parameters.
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask).to(tl.float32)

    ew = tl.exp(w)

    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask).to(tl.float32)

        euk = tl.exp(u + kt)
        wkv = (alpha + euk * vt) / (beta + euk)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)

        ek = tl.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c, alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c, beta, mask=cmask)


def wkv_triton_vanilla_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    (bsz, tsz, chans), device = k.shape, k.device

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 2, 1, chans), f"{state.shape} != {(bsz, 2, 1, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"

    # Checks tensor devices.
    for t in (v, state, w, u):
        assert t.device == device, f"{t.device} != {device}"

    # New tensors to output.
    wkvs = k.new_empty(bsz, tsz, chans)
    state_out = k.new_empty(bsz, 2, tsz, chans)

    # Constants.
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(chans, meta["BLOCK_SIZE_C"]))

    wkv_triton_vanilla_forward_kernel[grid](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(3),
        # WKV
        wkvs,
        wkvs.stride(0),
        wkvs.stride(1),
        wkvs.stride(2),
        # Output state
        state_out,
        state_out.stride(0),
        state_out.stride(1),
        state_out.stride(2),
        state_out.stride(3),
        # Params
        chans,
        tsz,
        BLOCK_SIZE_C=block_size_c,
    )

    state_out = torch.cat((state, state_out), dim=2)

    return wkvs, state_out


# @triton.autotune(configs=AUTOTUNE_CONFIGS, key=["chans"])
@triton.jit
def wkv_triton_vanilla_backward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_ab,
    state_s_t,
    state_s_c,
    # WKV grad
    gwkv_ptr,
    gwkv_s_b,
    gwkv_s_t,
    gwkv_s_c,
    # Output state grad
    gstate_out_ptr,
    gstate_out_s_b,
    gstate_out_s_ab,
    gstate_out_s_c,
    # W grad
    gw_ptr,
    gw_s_c,
    # U grad
    gu_ptr,
    gu_s_c,
    # K grad
    gk_ptr,
    gk_s_b,
    gk_s_t,
    gk_s_c,
    # V grad
    gv_ptr,
    gv_s_b,
    gv_s_t,
    gv_s_c,
    # State grad
    gstate_ptr,
    gstate_s_b,
    gstate_s_ab,
    gstate_s_c,
    # Params
    tsz,
    chans,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    cs = (c_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_ab

    # Pointers to the batch (and possibly channel) for the output tensors.
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b

    # Pointers to gradients which were recieved by the function.
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_ab

    # Loads parameters.
    galpha = tl.load(galpha_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.float32)
    gbeta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask).to(tl.float32)

    ew = tl.exp(w)

    # Gradient accumulators.
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)

    for t in range(tsz):
        tc = tsz - t - 1

        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask).to(tl.float32)

        alpha_prev = tl.load(alpha_ptr + tc * state_s_t + state_s_c * cs, mask=cmask).to(tl.float32)
        beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs, mask=cmask).to(tl.float32)

        euk = tl.exp(u + kt)
        ek = tl.exp(kt)

        denom = beta_prev + euk
        denom_sq = denom * denom

        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask).to(tl.float32)

        # Backpropagates wkv gradients.
        guk = gwkvt * euk * (beta_prev * vt - alpha_prev) / denom_sq
        gu += guk
        gk = guk
        gv = gwkvt * euk / denom

        galpha_wkv = gwkvt / denom
        gbeta_wkv = -gwkvt * (euk * vt + alpha_prev) / denom_sq

        # Backpropagates alpha gradients.
        gw += galpha * ew * alpha_prev
        gk += galpha * ek * vt
        gv += galpha * ek

        # Backpropagates beta gradients.
        gw += gbeta * ew * beta_prev
        gk += gbeta * ek

        # Stores the gradients for k and v.
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)

        # Computes new gradients for alpha and beta.
        galpha = galpha * ew + galpha_wkv
        gbeta = gbeta * ew + gbeta_wkv

    # Stores final gradients for alpha and beta.
    galpha_ptr = gstate_ptr + b_idx * gstate_s_b
    gbeta_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_ab
    tl.store(galpha_ptr + gstate_s_c * cs, galpha, mask=cmask)
    tl.store(gbeta_ptr + gstate_s_c * cs, gbeta, mask=cmask)

    # Stores final gradients for w and u.
    tl.atomic_add(gw_ptr + gw_s_c * cs, gw, mask=cmask)
    tl.atomic_add(gu_ptr + gu_s_c * cs, gu, mask=cmask)


def wkv_triton_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (bsz, tsz, chans), device = k.shape, k.device

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 2, tsz, chans), f"{state.shape} != {(bsz, 2, tsz, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 2, 1, chans)

    # Checks tensor devices.
    for t in (v, state, w, u, grad_wkv, grad_state):
        assert t.device == device, f"{t.device} != {device}"

    # New tensors to output.
    gw = torch.zeros_like(w)
    gu = torch.zeros_like(u)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gstate = k.new_empty(bsz, 2, 1, chans)

    # Constants.
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(chans, meta["BLOCK_SIZE_C"]))

    wkv_triton_vanilla_backward_kernel[grid](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        # WKV grad
        grad_wkv,
        grad_wkv.stride(0),
        grad_wkv.stride(1),
        grad_wkv.stride(2),
        # Output state grad
        grad_state,
        grad_state.stride(0),
        grad_state.stride(1),
        grad_state.stride(3),
        # W grad
        gw,
        gw.stride(0),
        # U grad
        gu,
        gu.stride(0),
        # K grad
        gk,
        gk.stride(0),
        gk.stride(1),
        gk.stride(2),
        # V grad
        gv,
        gv.stride(0),
        gv.stride(1),
        gv.stride(2),
        # State grad
        gstate,
        gstate.stride(0),
        gstate.stride(1),
        gstate.stride(3),
        # Params
        tsz,
        chans,
        BLOCK_SIZE_C=block_size_c,
    )

    return gw, gu, gk, gv, gstate


class WKVTritonFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        gw, gu, gk, gv, gstate = wkv_triton_vanilla_backward(w, u, k, v, state, gwkv, gstate)
        return gw, gu, gk, gv, gstate


def wkv_triton_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return WKVTritonFunction.apply(w, u, k, v, state)