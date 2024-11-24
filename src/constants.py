import os 

base_path = os.path.abspath(os.getcwd())
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
BASE_MODEL_CACHE = os.path.join( base_path, 'weights-cache/base_model')
CONTROLNET_MODEL = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
CONTROLNET_MODEL_CACHE = os.path.join( base_path, 'weights-cache/controlnet-inpainting-beta' ) 
CONTROL_LOCAL = ""
UPSCALER_CACHE = os.path.join( base_path, 'weights-cache/upscaler_model')
UPSCALER_MODEL = ""

hf_token = "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"