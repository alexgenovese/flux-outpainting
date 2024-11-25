import os, torch, time, shutil
from huggingface_hub import login, hf_hub_download
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
    login( token = hf_token )


# loaded into base model
def cache_model():
    
        try: 
            if not os.path.exists(CONTROLNET_MODEL_CACHE):
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

                return controlnet, pipe

            else: 
                # This case is because is being called from predict function 
                controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=_torch, cache_dir=CONTROLNET_MODEL_CACHE)
                transformer = FluxTransformer2DModel.from_pretrained(
                    BASE_MODEL, subfolder='transformer', torch_dtype=_torch, cache_dir=BASE_MODEL_CACHE
                )
                pipe = FluxControlNetInpaintingPipeline.from_pretrained(
                    BASE_MODEL,
                    transformer=transformer,
                    controlnet=controlnet,
                    torch_dtype=_torch,
                    cache_dir=BASE_MODEL_CACHE
                )
                return controlnet, pipe


        except Exception as error:
            print("CONTROLNET_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(CONTROLNET_MODEL_CACHE)
            print("CONTROLNET_MODEL - Removed empty cache directory")
        
            return False


#
#   Not currently in USE
#
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
            print("UPSCALER - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(UPSCALER_CACHE)
            print("UPSCALER - Removed empty cache directory")


def download_weights(): 
    print("-----> Start caching models...")
    with tqdm(total=100, desc="Pushing weights in cache folder") as pbar:
        login_hf()
        pbar.update(25)

        controlnet, pipe = cache_model()
        pbar.update(75)

    print("-----> Caching completed!")
    return controlnet, pipe


if __name__ == "__main__":
    download_weights() 