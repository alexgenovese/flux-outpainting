# COG Flux Outpainting + Hyper + Controlnet Inpainting 
It's a COG implementation of Flux Dev 1 + Hyper and Inpainting ControlNet checkpoint

## Flux.1-dev
**Licensing and commercial use**

If you generate images on Replicate with FLUX.1 models and their fine-tunes, then you can use the images commercially.

## Controlnet Inpainting Usage
Using t5xxl-FP16 and flux1-dev-fp8 models for 30-step inference @1024px & H20 GPU:

**GPU memory usage: 27GB**
Inference time: 48 seconds (true_cfg=3.5), 26 seconds (true_cfg=1)
Different results can be achieved by adjusting the following parameters:

**Parameter	Recommended Range Effect**

- control-strength	0.6 - 1.0	Controls how much influence the ControlNet has on the generation. Higher values result in stronger adherence to the control image.

- controlend-percent	0.35 - 1.0	Determines at which step in the denoising process the ControlNet influence ends. Lower values allow for more creative freedom in later steps.

- true-cfg (Classifier-Free Guidance Scale)	1.0 or 3.5	Influences how closely the generation follows the prompt. Higher values increase prompt adherence but may reduce image quality.


## Credits 
- https://huggingface.co/spaces/multimodalart/flux-outpainting @multimodalart
- Controlnet Inpainting Beta [AlimamaCreative Team](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)
- [ByteDance Flux Hyper](https://huggingface.co/ByteDance/Hyper-SD)