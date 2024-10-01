import os, torch, time, shutil
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from huggingface_hub import login, hf_hub_download
from diffusers.models.model_loading_utils import load_state_dict
from diffusers import AutoencoderKL, FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from constants import hf_token, VAE_CACHE, VAE_MODEL, BASE_MODEL, BASE_MODEL_CACHE, CONTROLNET_MODEL, CONTROLNET_MODEL_CACHE, UPSCALER_CACHE, UPSCALER_MODEL
from tqdm import tqdm

pipe = None
vae = None
logged_in = False 

def login_hf():
    if logged_in is False:
        login( token = hf_token )
    
    return True


def cache_vae():
    if not os.path.exists(VAE_CACHE): 
        try:
            os.makedirs(VAE_CACHE)
            vae = AutoencoderKL.from_pretrained(VAE_MODEL)
            vae.save_pretrained(VAE_CACHE, safe_serialization=True)
        except Exception as error:
            print("VAE - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(VAE_CACHE)
            print("VAE - Removed empty cache directory")

# loaded into base model
def cache_controlnet():
    if not os.path.exists(CONTROLNET_MODEL_CACHE):
        try: 
            os.makedirs(CONTROLNET_MODEL_CACHE)
            start = time.time()

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

            model.save_pretrained(CONTROLNET_MODEL_CACHE, safe_serialization=True)
            print("Downloading took: ", time.time() - start)

            return model

        except Exception as error:
            print("CONTROLNET_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(CONTROLNET_MODEL_CACHE)
            print("CONTROLNET_MODEL - Removed empty cache directory")


def cache_base_model():
    if not os.path.exists(BASE_MODEL_CACHE):
        try:
            os.makedirs(BASE_MODEL_CACHE)
            start = time.time()
            local_controlnet_union = cache_controlnet()

            pipe = StableDiffusionXLFillPipeline.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                vae=vae,
                controlnet=local_controlnet_union,
                variant="fp16",
                cache_dir=BASE_MODEL_CACHE
            )
            pipe.save_pretrained(BASE_MODEL_CACHE, safe_serialization=True)
            print("Downloading took: ", time.time() - start)
        except Exception as error:
            print("BASE_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(BASE_MODEL_CACHE)
            print("BASE_MODEL - Removed empty cache directory")


def cache_upscaler():
    if not os.path.exists(UPSCALER_CACHE): 
        try:
            os.makedirs(UPSCALER_CACHE)
            controlnet = FluxControlNetModel.from_pretrained(UPSCALER_MODEL)
            pipe = FluxControlNetPipeline.from_pretrained(
                BASE_MODEL, controlnet=controlnet
            )
            vae.save_pretrained(UPSCALER_CACHE, safe_serialization=True)
        except Exception as error:
            print("VAE - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(UPSCALER_CACHE)
            print("VAE - Removed empty cache directory")


def download_weights(): 
    print("-----> Start caching models...")
    with tqdm(total=100, desc="Creating cache") as pbar:
        login_hf()
        pbar.update(25)

        cache_vae()
        pbar.update(25)

        cache_base_model()
        pbar.update(25)

        cache_upscaler()
        pbar.update(25)

    print("-----> Caching completed!")


if __name__ == "__main__":
    download_weights() 