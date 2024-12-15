# ComfyUI wrapper nodes for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)

# ComfyUI-HunyuanVideoWrapper

This repository provides a wrapper for integrating HunyuanVideo into ComfyUI, allowing you to generate high-quality video content using advanced AI models.

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- `git-lfs`
- `cbm`
- `ffmpeg`

You can install these prerequisites using the following command:

```bash
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg
```

### Installation Steps

1. **Install `comfy-cli`:**

   ```bash
   pip install comfy-cli
   ```

2. **Initialize ComfyUI:**

   ```bash
   comfy --here install
   ```

3. **Clone and Install ComfyScript:**

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Chaoses-Ib/ComfyScript.git
   cd ComfyScript
   pip install -e ".[default,cli]"
   pip uninstall aiohttp
   pip install -U aiohttp
   ```

4. **Clone and Install ComfyUI-HunyuanVideoWrapper:**

   ```bash
   cd ../
   git clone https://github.com/svjack/ComfyUI-HunyuanVideoWrapper
   cd ComfyUI-HunyuanVideoWrapper
   pip install -r requirements.txt
   ```

5. **Load ComfyScript Runtime:**

   ```python
   from comfy_script.runtime import *
   load()
   from comfy_script.runtime.nodes import *
   ```

6. **Install Example Dependencies:**

   ```bash
   cd examples
   comfy node install-deps --workflow=hyvideo_t2v_example_01.json
   ```

7. **Update ComfyUI Dependencies:**

   ```bash
   cd ../../ComfyUI
   pip install --upgrade torch torchvision torchaudio -r requirements.txt
   ```

8. **Transpile Example Workflow:**

   ```bash
   python -m comfy_script.transpile hyvideo_t2v_example_01.json
   ```

9. **Download and Place Model Files:**

   Download the required model files from Hugging Face:

   ```bash
   huggingface-cli download Kijai/HunyuanVideo_comfy --local-dir ./HunyuanVideo_comfy
   ```

   Copy the downloaded files to the appropriate directories:

   ```bash
   cp -r HunyuanVideo_comfy/ .
   cp HunyuanVideo_comfy/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors ComfyUI/models/diffusion_models
   cp HunyuanVideo_comfy/hunyuan_video_vae_bf16.safetensors ComfyUI/models/vae
   ```

10. **Run the Example Script:**

    Create a Python script `run_t2v.py`:

    ```python
    from comfy_script.runtime import *
    load()
    from comfy_script.runtime.nodes import *
    with Workflow():
        vae = HyVideoVAELoader(r'hunyuan_video_vae_bf16.safetensors', 'bf16', None)
        model = HyVideoModelLoader(r'hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, None, None)
        hyvid_text_encoder = DownloadAndLoadHyVideoTextEncoder('Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'openai/clip-vit-large-patch14', 'fp16', False, 2, 'disabled')
        hyvid_embeds = HyVideoTextEncode(hyvid_text_encoder, '''high quality nature video of a red panda balancing on a bamboo stick while a bird lands on the panda's head, there's a waterfall in the background''', 'bad quality video', 'video', None, None, None)
        samples = HyVideoSampler(model, hyvid_embeds, 512, 320, 85, 30, 6, 9, 6, 1, None, 1, None)
        images = HyVideoDecode(vae, samples, True, 64, 256, True)
        _ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None,
                            pix_fmt = 'yuv420p', crf=19, save_metadata = True, trim_to_audio = False)
    ```

    Run the script:

    ```bash
    python run_t2v.py
    ```
<br/>

- prompt = "high quality nature video of a red panda balancing on a bamboo stick while a bird lands on the panda's head, there's a waterfall in the background"

https://github.com/user-attachments/assets/965ded95-7143-44b6-b125-b7b088aef0d9

<br/>

<br/>

- prompt = "high quality anime-style video of a chibi cat with big sparkling eyes, wearing a magical hat, holding a wand, and surrounded by glowing magical orbs, in a lush enchanted forest with floating cherry blossoms and a sparkling stream in the background"



https://github.com/user-attachments/assets/716954d9-d61f-406b-af9b-153c7bf62972



<br/>

## WORK IN PROGRESS

# Update

Scaled dot product attention (sdpa) should now be working (only tested on Windows, torch 2.5.1+cu124 on 4090), sageattention is still recommended for speed, but should not be necessary anymore making installation much easier.

Vid2vid test:
[source video](https://www.pexels.com/video/a-4x4-vehicle-speeding-on-a-dirt-road-during-a-competition-15604814/)

https://github.com/user-attachments/assets/12940721-4168-4e2b-8a71-31b4b0432314


text2vid (old test):

https://github.com/user-attachments/assets/3750da65-9753-4bd2-aae2-a688d2b86115


Transformer and VAE (single files, no autodownload):

https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main

Go to the usual ComfyUI folders (diffusion_models and vae)

LLM text encoder (has autodownload):

https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer

Files go to `ComfyUI/models/LLM/llava-llama-3-8b-text-encoder-tokenizer`

Clip text encoder (has autodownload)

Either use any Clip_L model supported by ComfyUI by disabling the clip_model in the text encoder loader and plugging in ClipLoader to the text encoder node, or 
allow the autodownloader to fetch the original clip model from:

https://huggingface.co/openai/clip-vit-large-patch14, (only need the .safetensor from the weights, and all the config files) to:

`ComfyUI/models/clip/clip-vit-large-patch14`

Memory use is entirely dependant on resolution and frame count, don't expect to be able to go very high even on 24GB. 

Good news is that the model can do functional videos even at really low resolutions.
