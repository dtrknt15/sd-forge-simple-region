import math
from functools import wraps

import torch
from torch.nn.functional import interpolate

from backend.nn.anima import SelfCrossAttention
from modules.devices import device, dtype

from .attention import _lcm_for_list


def _pad_seq_to_512(t: torch.Tensor) -> torch.Tensor:
    seq = t.shape[-2]
    target = math.ceil(seq / 512) * 512
    if target > seq:
        t = torch.nn.functional.pad(t, (0, 0, 0, target - seq))
    return t


def _get_dit_mask(
    mask: torch.Tensor, seq_len: int, w: int, h: int, patch_size: int = 2
) -> torch.Tensor:
    num_conds = mask.shape[0]
    h_p = max(h // (8 * patch_size), 1)
    w_p = max(w // (8 * patch_size), 1)
    spatial = h_p * w_p

    mask_for_interp = mask.view(num_conds, 1, mask.shape[-2], mask.shape[-1])
    mask_resized = interpolate(
        mask_for_interp, size=(h_p, w_p), mode="bilinear", align_corners=False
    )
    mask_flattened = mask_resized.reshape(num_conds, 1, spatial)

    reps = math.ceil(seq_len / spatial) if spatial else 1
    mask_flattened = mask_flattened.repeat(1, reps, 1)[:, :, :seq_len]
    return mask_flattened.reshape(num_conds, 1, seq_len, 1)


class SimpleAnimaCouple:
    """Apply masked cross-attention conditioning to Anima DiT models."""

    @staticmethod
    @torch.inference_mode()
    def patch_dit(
        model,
        cond_list: list[torch.Tensor],
        mask_list: list[torch.Tensor],
        width: int,
        height: int,
    ):
        if len(cond_list) < 2:
            return model

        SimpleAnimaCouple.unpatch()
        dit = model.model.diffusion_model

        mask = torch.stack(mask_list, dim=0).to(device=device, dtype=dtype)
        mask_sum = mask.sum(dim=0, keepdim=True)
        uncovered = mask_sum <= 1e-6
        if uncovered.any():
            mask[0] = torch.where(uncovered.squeeze(0), torch.ones_like(mask[0]), mask[0])
            mask_sum = mask.sum(dim=0, keepdim=True)
        mask = mask / mask_sum

        conds: list[torch.Tensor] = []
        for c in cond_list:
            c = c.to(device=device, dtype=dtype)
            if c.dim() == 3 and c.shape[1] == 1:
                c = c.squeeze(1)
            elif c.dim() == 3 and c.shape[0] == 1:
                c = c.squeeze(0)
            conds.append(_pad_seq_to_512(c))

        base_cond = conds[0]
        extra_conds = conds[1:]
        num_conds = len(conds)
        num_tokens = [c.shape[-2] for c in conds]

        SelfCrossAttention.sr_orig_forward = SelfCrossAttention.forward

        @wraps(SelfCrossAttention.sr_orig_forward)
        @torch.inference_mode()
        def sr_forward(
            self: "SelfCrossAttention",
            x: torch.Tensor,
            context: torch.Tensor,
            rope_emb: torch.Tensor,
            transformer_options: dict = {},
        ):
            if self.is_SelfAttn:
                return self.sr_orig_forward(
                    x=x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            cond_or_unconds = transformer_options.get("cond_or_uncond", None)
            if context is None or not cond_or_unconds:
                return self.sr_orig_forward(
                    x=x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            num_chunks = len(cond_or_unconds)
            batch_size = x.shape[0] // num_chunks
            x_chunks = x.chunk(num_chunks, dim=0)

            context_3d = context.squeeze(1)
            if context_3d.dim() == 3:
                seq = context_3d.shape[-2]
                target = math.ceil(seq / 512) * 512
                if target > seq:
                    context_3d = torch.nn.functional.pad(
                        context_3d, (0, 0, 0, target - seq)
                    )
            ctx_seq_len = context_3d.shape[-2]
            context_chunks = context_3d.chunk(num_chunks, dim=0)

            lcm_tokens = _lcm_for_list(num_tokens + [ctx_seq_len])
            if lcm_tokens > 1024:
                raise RuntimeError(
                    f"Simple Region (Anima): combined prompt token length LCM={lcm_tokens} "
                    "exceeds 1024. Shorten prompts or reduce the number of regions."
                )

            base_tensor = base_cond.repeat(batch_size, lcm_tokens // base_cond.shape[-2], 1)
            extra_tensor = torch.cat(
                [
                    c.repeat(batch_size, lcm_tokens // c.shape[-2], 1)
                    for c in extra_conds
                ],
                dim=0,
            )

            new_x = []
            new_context = []
            for idx, cond_or_uncond in enumerate(cond_or_unconds):
                c_target = context_chunks[idx].repeat(1, lcm_tokens // ctx_seq_len, 1)
                if cond_or_uncond == 1:
                    new_x.append(x_chunks[idx])
                    new_context.append(c_target)
                else:
                    new_x.append(x_chunks[idx].repeat(num_conds, 1, 1))
                    new_context.append(torch.cat([base_tensor, extra_tensor], dim=0))

            x_in = torch.cat(new_x, dim=0)
            ctx_in = torch.cat(new_context, dim=0).to(dtype=x_in.dtype)

            out = self.sr_orig_forward(
                x_in,
                context=ctx_in,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

            seq_len = out.shape[1]
            mask_down = _get_dit_mask(
                mask, seq_len, width, height, patch_size=dit.patch_spatial
            )

            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + num_conds * batch_size]
                    chunk = chunk.view(num_conds, batch_size, seq_len, -1)
                    outputs.append((chunk * mask_down).sum(dim=0))
                    pos += num_conds * batch_size

            return torch.cat(outputs, dim=0)

        sr_forward._simple_region = True
        SelfCrossAttention.forward = sr_forward
        return model

    @staticmethod
    def unpatch():
        if hasattr(SelfCrossAttention, "sr_orig_forward"):
            if getattr(SelfCrossAttention.forward, "_simple_region", False):
                SelfCrossAttention.forward = SelfCrossAttention.sr_orig_forward
            del SelfCrossAttention.sr_orig_forward
