import math
import torch
from torch.nn.functional import interpolate

from modules.devices import device, dtype


def _repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)
    return value


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _lcm_for_list(numbers: list[int]) -> int:
    current = numbers[0]
    for num in numbers[1:]:
        current = _lcm(current, num)
    return current


def _downsample_mask(
    mask: torch.Tensor, batch_size: int, num_tokens: int, shape: tuple[int, ...]
) -> torch.Tensor:
    width, height = shape[3], shape[2]
    scale = math.ceil(math.log2(math.sqrt(height * width / num_tokens)))
    size = (_repeat_div(height, scale), _repeat_div(width, scale))

    num_conds = mask.shape[0]
    mask_down = interpolate(mask, size=size, mode="nearest")

    spatial = mask_down.shape[-2] * mask_down.shape[-1]
    flat = mask_down.reshape(num_conds, spatial, 1)

    if spatial < num_tokens:
        pad = flat.new_zeros(num_conds, num_tokens - spatial, 1)
        flat = torch.cat([flat, pad], dim=1)
    elif spatial > num_tokens:
        flat = flat[:, :num_tokens, :]

    return flat.repeat_interleave(batch_size, dim=0)


class SimpleAttentionCouple:
    """Apply masked cross-attention conditioning to a Forge UNet clone."""

    @classmethod
    @torch.inference_mode()
    def patch_unet(
        cls,
        model,
        cond_list: list[torch.Tensor],
        mask_list: list[torch.Tensor],
    ):
        """
        cond_list: list of tensors shaped [1, tokens, dim] (index 0 is base/global prompt)
        mask_list: list of tensors shaped [1, H, W] matching cond_list
        """
        if len(cond_list) < 2:
            return model

        m = model.clone()

        mask = torch.stack(mask_list, dim=0).to(device=device, dtype=dtype)
        # Ensure no pixel has zero total weight to avoid division by zero / NaNs
        mask_sum = mask.sum(dim=0, keepdim=True)
        uncovered = mask_sum <= 1e-6
        if uncovered.any():
            mask[0] = torch.where(uncovered.squeeze(0), torch.ones_like(mask[0]), mask[0])
            mask_sum = mask.sum(dim=0, keepdim=True)

        mask = mask / mask_sum

        base_cond = cond_list[0].to(device=device, dtype=dtype)
        extra_conds = [c.to(device=device, dtype=dtype) for c in cond_list[1:]]
        num_conds = len(cond_list)
        extra_tokens = [c.shape[1] for c in extra_conds]

        state = {"batch_size": 1}

        @torch.inference_mode()
        def attn2_patch(q, k, v, extra_options):
            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            batch_size = q.shape[0] // num_chunks
            state["batch_size"] = batch_size

            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            lcm_tokens = _lcm_for_list(extra_tokens + [k.shape[1], base_cond.shape[1]])

            base_tensor = base_cond.repeat(batch_size, lcm_tokens // base_cond.shape[1], 1)
            extra_tensor = torch.cat(
                [
                    c.repeat(batch_size, lcm_tokens // extra_tokens[i], 1)
                    for i, c in enumerate(extra_conds)
                ],
                dim=0,
            )

            qs, ks = [], []
            for i, cond_or_uncond in enumerate(cond_or_unconds):
                k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
                if cond_or_uncond == 1:  # uncond
                    qs.append(q_chunks[i])
                    ks.append(k_target)
                else:
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([base_tensor, extra_tensor], dim=0))

            qs = torch.cat(qs, dim=0).to(q)
            ks = torch.cat(ks, dim=0).to(k)

            if qs.size(0) % 2 == 1:
                qs = torch.cat((qs, torch.zeros_like(qs[0]).unsqueeze(0)), dim=0)
                ks = torch.cat((ks, torch.zeros_like(ks[0]).unsqueeze(0)), dim=0)

            return qs, ks, ks

        @torch.inference_mode()
        def attn2_output_patch(out, extra_options):
            cond_or_unconds = extra_options["cond_or_uncond"]
            batch_size = state["batch_size"]
            mask_down = _downsample_mask(
                mask, batch_size, out.shape[1], extra_options["original_shape"]
            )

            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + num_conds * batch_size]
                    weighted = (chunk * mask_down).view(num_conds, batch_size, out.shape[1], -1)
                    outputs.append(weighted.sum(dim=0))
                    pos += num_conds * batch_size

            return torch.cat(outputs, dim=0)

        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)
        return m
