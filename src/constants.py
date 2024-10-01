import os 

base_path = os.path.abspath(os.getcwd())
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix" 
VAE_CACHE = os.path.join( base_path, 'weights-cache/vae' ) 
BASE_MODEL = "SG161222/RealVisXL_V5.0_Lightning"
BASE_MODEL_CACHE = os.path.join( base_path, 'weights-cache/base_model')
CONTROLNET_MODEL = "xinsir/controlnet-union-sdxl-1.0"
CONTROLNET_MODEL_CACHE = os.path.join( base_path, 'weights-cache/controlnet-union' ) 
CONTROL_LOCAL = ""
UPSCALER_CACHE = os.path.join( base_path, 'weights-cache/upscaler_model')
UPSCALER_MODEL = "jasperai/Flux.1-dev-Controlnet-Upscaler"

hf_token = "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"