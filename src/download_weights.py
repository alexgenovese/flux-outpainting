import os, torch, time, shutil
from huggingface_hub import login, hf_hub_download, login
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from diffusers.pipelines import FluxControlNetPipeline
from constants import hf_token, BASE_MODEL, BASE_MODEL_CACHE, CONTROLNET_MODEL, CONTROLNET_MODEL_CACHE, UPSCALER_CACHE, UPSCALER_MODEL
from tqdm import tqdm

pipe = None
vae = None
logged_in = False 
_torch = torch.float16

def login_hf():
    if logged_in is False:
        login( token = hf_token )
    
    return True


# loaded into base model
def cache_model():
    if not os.path.exists(CONTROLNET_MODEL_CACHE):
        try: 
            os.makedirs(CONTROLNET_MODEL_CACHE)
            start = time.time()

            controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=_torch, cache_dir=CONTROLNET_MODEL_CACHE)
            controlnet.save_pretrained(CONTROLNET_MODEL_CACHE)

            transformer = FluxTransformer2DModel.from_pretrained(
                BASE_MODEL, subfolder='transformer', torch_dtype=_torch, cache_dir=BASE_MODEL_CACHE
            )
            transformer.save_pretrained(BASE_MODEL_CACHE)

            pipe = FluxControlNetInpaintingPipeline.from_pretrained(
                BASE_MODEL,
                transformer=transformer,
                controlnet=controlnet,
                torch_dtype=_torch,
                cache_dir=BASE_MODEL_CACHE
            )
            pipe.save_pretrained(BASE_MODEL_CACHE)

            print("Downloading took: ", time.time() - start)

            return pipe

        except Exception as error:
            print("CONTROLNET_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(CONTROLNET_MODEL_CACHE)
            print("CONTROLNET_MODEL - Removed empty cache directory")



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

        cache_model()
        pbar.update(75)

    print("-----> Caching completed!")


if __name__ == "__main__":
    download_weights() 