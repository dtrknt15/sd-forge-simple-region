import json
import logging
from typing import TYPE_CHECKING

import gradio as gr
import torch

from lib_simple_region.attention import SimpleAttentionCouple
from modules import scripts
from modules.prompt_parser import SdConditioning

try:
    from lib_simple_region.anima import SimpleAnimaCouple

    HAS_ANIMA = True
except ImportError:
    SimpleAnimaCouple = None
    HAS_ANIMA = False

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing as P

logger = logging.getLogger("SimpleRegion")


def _is_anima(sd_model) -> bool:
    if not HAS_ANIMA:
        return False
    repo = getattr(getattr(sd_model, "model_config", None), "huggingface_repo", "") or ""
    return sd_model.__class__.__name__ == "Anima" or repo.endswith("Anima")


def _encode_prompt(sd_model, text: str, width: int, height: int) -> torch.Tensor:
    texts = SdConditioning([text], False, width, height, None)
    cond = sd_model.get_learned_conditioning(texts)
    if getattr(sd_model, "is_sdxl", False):
        return cond["crossattn"]
    if getattr(sd_model, "is_sd1", False):
        return cond
    if _is_anima(sd_model):
        return cond[0][0]
    raise NotImplementedError("Simple Region currently supports SDXL, SD1.5, and Anima models.")


def _parse_regions(regions_json: str) -> list[dict]:
    if not regions_json or not regions_json.strip():
        return []
    try:
        data = json.loads(regions_json)
        if not isinstance(data, list):
            return []
        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            x1 = max(0.0, min(1.0, float(item.get("x1", 0.0))))
            y1 = max(0.0, min(1.0, float(item.get("y1", 0.0))))
            x2 = max(0.0, min(1.0, float(item.get("x2", 1.0))))
            y2 = max(0.0, min(1.0, float(item.get("y2", 1.0))))
            if x2 <= x1 or y2 <= y1:
                continue
            weight = max(0.0, float(item.get("weight", 1.0)))
            valid.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "prompt": prompt,
                    "weight": weight,
                }
            )
        return valid
    except Exception as e:
        logger.warning(f"Failed to parse regions JSON: {e}")
        return []


class SimpleRegion(scripts.Script):
    def __init__(self):
        self.is_img2img = False
        self.is_hr = False
        self.base_prompt = ""

    def title(self):
        return "Simple Region"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.is_img2img = is_img2img
        tab = "i2i" if is_img2img else "t2i"

        i2i_bg_btn = (
            '<button type="button" class="sr-btn sr-btn-bg-i2i">i2iから取得</button>'
            if is_img2img
            else ""
        )

        with gr.Accordion("Simple Region", open=False, elem_id=f"sr_{tab}_accordion"):
            with gr.Row():
                enable = gr.Checkbox(
                    label="Enable",
                    value=False,
                    elem_id=f"sr_{tab}_enable",
                )
                inherit_prompt = gr.Checkbox(
                    label="通常プロンプトを領域にも引き継ぐ",
                    value=True,
                    elem_id=f"sr_{tab}_inherit",
                )
                base_weight = gr.Slider(
                    label="全体(背景)の重み",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    elem_id=f"sr_{tab}_base_weight",
                )

            regions_json = gr.Textbox(
                value="[]",
                visible=False,
                elem_id=f"sr_{tab}_regions_json",
            )

            gr.HTML(
                f"""
                <div id="sr_{tab}_root" class="sr-root" data-tab="{tab}">
                    <div class="sr-toolbar">
                        <button type="button" class="sr-btn sr-btn-add">+ 範囲追加</button>
                        <label class="sr-split-label">
                            分割数
                            <input type="number" class="sr-split-count" min="2" max="8" value="2" />
                        </label>
                        <button type="button" class="sr-btn sr-btn-split-h">横分割 (左右)</button>
                        <button type="button" class="sr-btn sr-btn-split-v">縦分割 (上下)</button>
                        <button type="button" class="sr-btn sr-btn-sync" title="WebUIのWidth/Heightを再取得してキャンバス比率を更新">解像度更新</button>
                        <button type="button" class="sr-btn sr-btn-danger sr-btn-clear">全消去</button>
                        <span class="sr-hint">キャンバス上をドラッグして範囲を追加</span>
                    </div>
                    <div class="sr-workspace">
                        <div class="sr-canvas-wrap">
                            <canvas class="sr-canvas" width="384" height="384"></canvas>
                            <div class="sr-size-label"></div>
                            <div class="sr-bg-controls">
                                <label class="sr-btn sr-btn-sm">
                                    背景画像
                                    <input type="file" accept="image/*" class="sr-bg-file" hidden />
                                </label>
                                {i2i_bg_btn}
                                <button type="button" class="sr-btn sr-btn-sm sr-btn-bg-clear">背景クリア</button>
                            </div>
                        </div>
                        <div class="sr-list"></div>
                    </div>
                </div>
                """,
                elem_id=f"sr_{tab}_html",
            )

        return [enable, inherit_prompt, base_weight, regions_json]

    def setup(self, p: "P", *args, **kwargs):
        self.is_hr = False

    def before_hr(self, p: "P", *args, **kwargs):
        self.is_hr = True
        if HAS_ANIMA and _is_anima(p.sd_model):
            SimpleAnimaCouple.unpatch()

    def before_process_batch(self, p: "P", *args, **kwargs):
        if HAS_ANIMA and _is_anima(p.sd_model):
            SimpleAnimaCouple.unpatch()

    def postprocess(self, p: "P", processed, *args):
        if HAS_ANIMA:
            SimpleAnimaCouple.unpatch()

    def after_extra_networks_activate(
        self,
        p: "P",
        enable: bool,
        inherit_prompt: bool,
        base_weight: float,
        regions_json: str,
        *args,
        **kwargs,
    ):
        self.base_prompt = ""
        if not enable:
            return

        prompts = kwargs.get("prompts") or getattr(p, "all_prompts", None) or [p.prompt]
        self.base_prompt = prompts[0] if prompts else ""

        regions = _parse_regions(regions_json)
        if not regions:
            return

        p.extra_generation_params.update(
            {
                "SimpleRegion_Enable": True,
                "SimpleRegion_Inherit": inherit_prompt,
                "SimpleRegion_BaseWeight": base_weight,
                "SimpleRegion_Regions": json.dumps(regions, ensure_ascii=False),
            }
        )

    def process_before_every_sampling(
        self,
        p: "P",
        enable: bool,
        inherit_prompt: bool,
        base_weight: float,
        regions_json: str,
        *args,
        **kwargs,
    ):
        if not enable:
            return

        if getattr(p, "_ad_inner", False):
            return

        is_anima = _is_anima(p.sd_model)
        if not (
            getattr(p.sd_model, "is_sdxl", False)
            or getattr(p.sd_model, "is_sd1", False)
            or is_anima
        ):
            logger.warning(
                "Simple Region currently supports SDXL / SD1.5 / Anima models only. Skipping."
            )
            return

        regions = _parse_regions(regions_json)
        if not regions:
            return

        width = int(p.hr_upscale_to_x if self.is_hr else p.width)
        height = int(p.hr_upscale_to_y if self.is_hr else p.height)

        base_prompt = self.base_prompt or p.prompt or ""
        cond_list: list[torch.Tensor] = [
            _encode_prompt(p.sd_model, base_prompt, width, height)
        ]
        mask_list: list[torch.Tensor] = [
            torch.ones((1, height, width)) * max(0.0, float(base_weight))
        ]

        for reg in regions:
            if inherit_prompt and base_prompt.strip():
                combined_prompt = f"{base_prompt.strip()}, {reg['prompt']}"
            else:
                combined_prompt = reg["prompt"]

            cond = _encode_prompt(p.sd_model, combined_prompt, width, height)

            x1 = int(round(width * reg["x1"]))
            x2 = int(round(width * reg["x2"]))
            y1 = int(round(height * reg["y1"]))
            y2 = int(round(height * reg["y2"]))

            mask = torch.zeros((1, height, width))
            mask[:, y1:y2, x1:x2] = reg["weight"]

            cond_list.append(cond)
            mask_list.append(mask)

        unet = p.sd_model.forge_objects.unet
        if is_anima:
            p.sd_model.forge_objects.unet = SimpleAnimaCouple.patch_dit(
                unet, cond_list, mask_list, width, height
            )
        else:
            p.sd_model.forge_objects.unet = SimpleAttentionCouple.patch_unet(
                unet, cond_list, mask_list
            )
