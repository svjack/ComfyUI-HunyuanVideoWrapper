# ComfyUI wrapper nodes for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)

## WORK IN PROGRESS

Transformer and VAE (single files, no autodownload):

https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main

Go to the usual ComfyUI folders (diffusion_models and vae)

LLM text encoder (has autodownload):

https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer

Files go to `ComfyUI/models/LLM/llava-llama-3-8b-text-encoder-tokenizer`

Clip text encoder (has autodownload)

For now using the original https://huggingface.co/openai/clip-vit-large-patch14, files (only need the .safetensor from the weights) go to:

`ComfyUI/models/clip/clip-vit-large-patch14`