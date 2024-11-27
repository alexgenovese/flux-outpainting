import os 
base_path = os.path.join(os.path.abspath(os.getcwd()), 'weights-cache')

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = base_path
os.environ["TORCH_HOME"] = base_path
os.environ["HF_DATASETS_CACHE"] = base_path
os.environ["TRANSFORMERS_CACHE"] = base_path
os.environ["HUGGINGFACE_HUB_CACHE"] = base_path


BASE_MODEL = "black-forest-labs/FLUX.1-dev"
BASE_MODEL_CACHE = os.path.join( base_path, 'base_model')
CONTROLNET_MODEL = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
CONTROLNET_MODEL_CACHE = os.path.join( base_path, 'controlnet-inpainting-beta' ) 
CONTROL_LOCAL = ""
UPSCALER_CACHE = os.path.join( base_path, 'upscaler_model')
UPSCALER_MODEL = ""

hf_token = "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"