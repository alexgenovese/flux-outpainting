import torch, os, sys
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw

sys.path.append("./src")
from src.utils import get_torch_device
from src.download_weights import download_weights
from src.constants import hf_token, VAE_CACHE, VAE_MODEL, BASE_MODEL, BASE_MODEL_CACHE, CONTROLNET_MODEL, CONTROLNET_MODEL_CACHE, base_path

from diffusers.utils import load_image


class Predictor(BasePredictor):

    def setup(self):
        # Download or cache
        download_weights()

        config_file = hf_hub_download(
            CONTROLNET_MODEL,
            filename="config_promax.json",
            cache_dir=CONTROLNET_MODEL_CACHE
        )

        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        model_file = hf_hub_download(
            CONTROLNET_MODEL,
            filename="diffusion_pytorch_model_promax.safetensors",
            cache_dir=CONTROLNET_MODEL_CACHE
        )
        state_dict = load_state_dict(model_file)
        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, CONTROLNET_MODEL,
        )
        model.to(device=get_torch_device(), dtype=torch.float16)

        vae = AutoencoderKL.from_pretrained(
            VAE_MODEL, torch_dtype=torch.float16, 
            cache_dir=VAE_CACHE
        ).to(get_torch_device())

        self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
            cache_dir=BASE_MODEL_CACHE
        ).to(get_torch_device())

        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

    
    def predict(self, 
                image: Path = Input(description="Image", default=None), 
                width: int = Input(description="Width", default=720), 
                height: int = Input(description="height", default=1280), 
                overlap_width: int = Input(description="overlap width", default=72), 
                num_inference_steps: int = Input(description="Steps", default=8),
                resize_option: str = Input(description="height", default="Full"), 
                custom_resize_size: str = Input(description="Custom Resize Size (Optional)", default=""), 
                prompt_input: str = Input(
                    description="Write here your prompt (Optional)",
                    default=""
                ),
                alignment: str = Input(
                    description="Alignment",
                    default="Middle"
                )
        ) -> str:
        init_image = load_image( image )
        init_image.convert("RGB")

        source = init_image
        target_size = (width, height)
        overlap = overlap_width

        # Upscale if source is smaller than target in both dimensions
        if source.width < target_size[0] and source.height < target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = int(source.width * scale_factor)
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)

        if source.width > target_size[0] or source.height > target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = int(source.width * scale_factor)
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)
        
        if resize_option == "Full":
            resize_size = max(source.width, source.height)
        elif resize_option == "1/2":
            resize_size = max(source.width, source.height) // 2
        elif resize_option == "1/3":
            resize_size = max(source.width, source.height) // 3
        elif resize_option == "1/4":
            resize_size = max(source.width, source.height) // 4
        else:  # Custom
            resize_size = custom_resize_size

        aspect_ratio = source.height / source.width
        new_width = resize_size
        new_height = int(resize_size * aspect_ratio)
        source = source.resize((new_width, new_height), Image.LANCZOS)

        if not self.can_expand(source.width, source.height, target_size[0], target_size[1], alignment):
            alignment = "Middle"

        # Calculate margins based on alignment
        if alignment == "Middle":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - source.width
            margin_y = (target_size[1] - source.height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - source.width) // 2
            margin_y = target_size[1] - source.height

        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)

        # Adjust mask generation based on alignment
        if alignment == "Middle":
            mask_draw.rectangle([
                (margin_x + overlap, margin_y + overlap),
                (margin_x + source.width - overlap, margin_y + source.height - overlap)
            ], fill=0)
        elif alignment == "Left":
            mask_draw.rectangle([
                (margin_x, margin_y),
                (margin_x + source.width - overlap, margin_y + source.height)
            ], fill=0)
        elif alignment == "Right":
            mask_draw.rectangle([
                (margin_x + overlap, margin_y),
                (margin_x + source.width, margin_y + source.height)
            ], fill=0)
        elif alignment == "Top":
            mask_draw.rectangle([
                (margin_x, margin_y),
                (margin_x + source.width, margin_y + source.height - overlap)
            ], fill=0)
        elif alignment == "Bottom":
            mask_draw.rectangle([
                (margin_x, margin_y + overlap),
                (margin_x + source.width, margin_y + source.height)
            ], fill=0)

        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        try: 
            final_prompt = f"{prompt_input} , high quality, 4k, 8k, high resolution, detailed"

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(final_prompt, get_torch_device(), True)

            for image in self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                image=cnet_image,
                num_inference_steps=num_inference_steps
            ):
                # yield cnet_image, image
                pass

            image = image.convert("RGBA")
            cnet_image.paste(image, (0, 0), mask)
        except Exception as error:
            print("----- Something went wrong")
            print(f"{error}")

        # yield background, cnet_image
        cnet_image.save('./image.png')
        # background.save('./background.png')
        return './image.png'


    def can_expand(self, source_width, source_height, target_width, target_height, alignment):
        """Checks if the image can be expanded based on the alignment."""
        if alignment in ("Left", "Right") and source_width >= target_width:
            return False
        if alignment in ("Top", "Bottom") and source_height >= target_height:
            return False
        return True
