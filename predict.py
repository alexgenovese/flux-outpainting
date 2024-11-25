import torch, os, sys
import numpy as np
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from PIL import Image, ImageDraw

sys.path.append("./src")
from src.controlnet_flux import FluxControlNetModel
from src.transformer_flux import FluxTransformer2DModel
from src.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from src.utils import get_torch_device
from src.download_weights import download_weights
from src.constants import hf_token, BASE_MODEL, BASE_MODEL_CACHE, CONTROLNET_MODEL, CONTROLNET_MODEL_CACHE, base_path

_torch = torch.bfloat16
# _torch = torch.float16

class Predictor(BasePredictor):
    def setup(self):
        print("Setup - Download or get weights from cache")
        # Download or get the weights from cache
        controlnet, pipe = download_weights()

        self.controlnet = controlnet
        self.pipe = pipe
        print("Setup - Completed")

    
    def predict(self, 
            image: Path = Input(description="Image", default=None), 
            width: int = Input(description="Width", default=720), 
            height: int = Input(description="height", default=1280), 
            overlap_width: int = Input(description="overlap width", default=72), 
            num_inference_steps: int = Input(description="Steps", default=8),
            resize_option: str = Input(
                description="Zoom out (Optional) - Full: no zoom", 
                default="Full",
                choices=["Full", "50%", "33%", "25%"]
            ),
            custom_resize_size: str = Input(description="Percentage of resize", default="50"), 
            prompt_input: str = Input(
                description="Write here your prompt (Optional)",
                default=""
            ),
            alignment: str = Input(
                description="Alignment",
                default="Middle",
                choices=["Top", "Middle", "Left", "Right", "Bottom"]
            )
        ) -> Path:
        print("Predict - Start inference")
        init_image = Image.open(image)
        init_image.convert("RGB")

        custom_resize_percentage = custom_resize_size
        overlap_left=8
        overlap_right=8
        overlap_bottom=8
        overlap_top=8

        print("Predict - Prepare image and mask")
        background, mask = self.prepare_image_and_mask(image, width, height, overlap_width, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)

        print("Predict - Can Expand")
        if not self.can_expand(background.width, background.height, width, height, alignment):
            alignment = "Middle"

        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        final_prompt = f"{prompt_input} , high quality, 4k, 8k, high resolution, detailed skin, details"

        #generator = torch.Generator(device="cuda").manual_seed(42)

        try: 
            print("Predict - Adding Hyper")
            self.pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"))
            self.pipe.fuse_lora(lora_scale=0.125)
            self.pipe.transformer.to(torch.bfloat16)
            self.pipe.controlnet.to(torch.bfloat16)
            self.pipe.to(get_torch_device())

            print("Predict - Run Inference pipe")
            result = self.pipe(
                prompt=final_prompt,
                height=height,
                width=width,
                control_image=cnet_image,
                control_mask=mask,
                num_inference_steps=num_inference_steps,
                #generator=generator,
                controlnet_conditioning_scale=0.9,
                guidance_scale=3.5,
                negative_prompt="",
                true_guidance_scale=3.5,
            ).images[0]

            print("Predict - Manipulating image and mask")
            result = result.convert("RGBA")
            cnet_image.paste(result, (0, 0), mask)

            # yield background, cnet_image
            print("Predict - Save locally")
            cnet_image.save('/tmp/flux-outpainted-image.png')
            # return cnet_image, background

            print("Predict - Print the path")
            return Path('/tmp/flux-outpainted-image.png')

        except Exception as error:
            print("----- Something went wrong")
            print(f"{error}")
            return False
        


    def can_expand(self, source_width, source_height, target_width, target_height, alignment):
        """Checks if the image can be expanded based on the alignment."""
        if alignment in ("Left", "Right") and source_width >= target_width:
            return False
        if alignment in ("Top", "Bottom") and source_height >= target_height:
            return False
        return True


    def prepare_image_and_mask(self, image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
        target_size = (width, height)

        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Resize the source image to fit within target size
        source = image.resize((new_width, new_height), Image.LANCZOS)

        # Apply resize option using percentages
        if resize_option == "Full":
            resize_percentage = 100
        elif resize_option == "50%":
            resize_percentage = 50
        elif resize_option == "33%":
            resize_percentage = 33
        elif resize_option == "25%":
            resize_percentage = 25
        else:  # Custom
            resize_percentage = custom_resize_percentage

        # Calculate new dimensions based on percentage
        resize_factor = resize_percentage / 100
        new_width = int(source.width * resize_factor)
        new_height = int(source.height * resize_factor)

        # Ensure minimum size of 64 pixels
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)

        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the overlap in pixels based on the percentage
        overlap_x = int(new_width * (overlap_percentage / 100))
        overlap_y = int(new_height * (overlap_percentage / 100))

        # Ensure minimum overlap of 1 pixel
        overlap_x = max(overlap_x, 1)
        overlap_y = max(overlap_y, 1)

        # Calculate margins based on alignment
        if alignment == "Middle":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - new_width
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = target_size[1] - new_height

        # Adjust margins to eliminate gaps
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))

        # Create a new background image and paste the resized source image
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)

        # Calculate overlap areas
        white_gaps_patch = 2

        left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
        
        if alignment == "Left":
            left_overlap = margin_x + overlap_x if overlap_left else margin_x
        elif alignment == "Right":
            right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
        elif alignment == "Top":
            top_overlap = margin_y + overlap_y if overlap_top else margin_y
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

        # Draw the mask
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)

        return background, mask

if __name__ == "__main__":
    pred = Predictor()
    pred.setup()
    pred.predict()