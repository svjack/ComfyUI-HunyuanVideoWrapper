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
   git clone https://github.com/kijai/ComfyUI-HunyuanVideoWrapper
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

### HunyuanCustom 
```bash
cp HunyuanVideo_comfy/hunyuan_video_custom_720p_fp8_scaled.safetensors ComfyUI/models/diffusion_models

wget https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/clip_vision/llava_llama3_vision.safetensors

cp llava_llama3_vision.safetensors ComfyUI/models/clip_vision

wget https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors

cp llava_llama3_fp8_scaled.safetensors ComfyUI/models/clip

wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors

cp clip_l.safetensors ComfyUI/models/clip

pip install sageattention

comfy launch -- --listen 0.0.0.0
```


# ComfyUI-HunyuanVideoWrapper - Lora Integration

This repository extends the functionality of **ComfyUI-HunyuanVideoWrapper** by adding support for **Lora** models, enabling the generation of high-quality video content with custom character and action LoRA models.

## Lora Integration

### Overview

Open source is truly amazing, and **HunyuanVideo** now supports LoRA models! I recently tested **HunyuanVideo** with both **action LoRA** and **character LoRA**, and the results are fantastic.

### Resources

- **Repository:** [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)
- **Workflow Example:** [HunyuanVideo LoRA Workflow](https://github.com/comfyonline/comfyonline_workflow/blob/main/hunyuanvideo%20lora%20Walking%20Animation%20Share.json)

### Installation

#### Step-by-Step Guide

1. **Download the LoRA Model:**

   Download the LoRA model from CivitAI:

   - [Walking Animation LoRA](https://civitai.com/models/1032126/walking-animation-hunyuan-video?modelVersionId=1157591)

   ```bash
   kxsr_walking_anim_v1-5.safetensors
   ```

   Copy the model to the `loras` directory:

   ```bash
   cp kxsr_walking_anim_v1-5.safetensors ComfyUI/models/loras
   ```

2. **Install Workflow Dependencies:**

   Install dependencies for the workflow:

   ```bash
   comfy node install-deps --workflow='hunyuanvideo lora Walking Animation Share.json'
   ```

3. **Transpile the Workflow:**

   Transpile the workflow file:

   ```bash
   python -m comfy_script.transpile 'hunyuanvideo lora Walking Animation Share.json'
   ```

4. **Run the Workflow:**

   Create a Python script `run_t2v_walking_lora.py`:

   ```python
   from comfy_script.runtime import *
   load()
   from comfy_script.runtime.nodes import *
   with Workflow():
       vae = HyVideoVAELoader(r'hunyuan_video_vae_bf16.safetensors', 'bf16', None)
       lora = HyVideoLoraSelect('kxsr_walking_anim_v1-5.safetensors', 1, None, None)
       model = HyVideoModelLoader(r'hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, None, lora)
       hyvid_text_encoder = DownloadAndLoadHyVideoTextEncoder('Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'openai/clip-vit-large-patch14', 'fp16', False, 2, 'disabled')
       hyvid_embeds = HyVideoTextEncode(hyvid_text_encoder, "kxsr, Shrek, full body, no_crop", 'bad quality video', 'video', None, None, None)
       samples = HyVideoSampler(model, hyvid_embeds, 512, 320, 85, 30, 6, 9, 6, 1, None, 1, None)
       images = HyVideoDecode(vae, samples, True, 64, 256, True)
       _ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None,
                           pix_fmt = 'yuv420p', crf=19, save_metadata = True, trim_to_audio = False)
   ```

   Run the script:

   ```bash
   python run_t2v_walking_lora.py
   ```

<br/>

- prompt = "kxsr, Shrek, full body, no_crop"



https://github.com/user-attachments/assets/47dba483-a113-4872-a4b4-e6ac6098967b


<br/>

### Online Demo

- **Action LoRA Demo:** [ComfyOnline - Action LoRA](https://www.comfyonline.app/explore/48aa3381-e9f7-4e16-8e41-96ff4faca263)
- **Character LoRA Demo:** [ComfyOnline - Character LoRA](https://www.comfyonline.app/explore/65940ef3-fcde-415b-a27f-ca71cd82d6ab)

### Makima Character LoRA Example

1. **Download the Makima LoRA Model:**

   Download the Makima LoRA model from CivitAI:

   - [Makima Hunyuan Character LoRA](https://civitai.com/models/1029279/makima-hunyuan-character?modelVersionId=1154429)

   ```bash
   makima_hunyuan.safetensors
   ```

   Copy the model to the `loras` directory:

   ```bash
   cp makima_hunyuan.safetensors ComfyUI/models/loras
   ```

2. **Install Workflow Dependencies:**

   Install dependencies for the workflow:

   ```bash
   comfy node install-deps --workflow='hunyuan video lora makima character.json'
   ```

3. **Transpile the Workflow:**

   Transpile the workflow file:

   ```bash
   python -m comfy_script.transpile 'hunyuan video lora makima character.json'
   ```

4. **Run the Workflow:**

   Create a Python script `run_t2v_makima_lora.py`:

   ```python
   from comfy_script.runtime import *
   load()
   from comfy_script.runtime.nodes import *
   with Workflow():
       vae = HyVideoVAELoader(r'hunyuan_video_vae_bf16.safetensors', 'bf16', None)
       lora = HyVideoLoraSelect('makima_hunyuan.safetensors', 1, None, None)
       model = HyVideoModelLoader(r'hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, None, lora)
       hyvid_text_encoder = DownloadAndLoadHyVideoTextEncoder('Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'openai/clip-vit-large-patch14', 'fp16', False, 2, 'disabled')
       hyvid_embeds = HyVideoTextEncode(hyvid_text_encoder, "kxsr, 1 lively kxsr running on campus， cinematic, anime aesthetic", 'bad quality video', 'video', None, None, None)
       samples = HyVideoSampler(model, hyvid_embeds, 512, 320, 85, 30, 6, 9, 6, 1, None, 1, None)
       images = HyVideoDecode(vae, samples, True, 64, 256, True)
       _ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None,
                           pix_fmt = 'yuv420p', crf=19, save_metadata = True, trim_to_audio = False)
   ```

   Run the script:

   ```bash
   python run_t2v_makima_lora.py
   ```

<br/>

- prompt = "kxsr, 1 lively kxsr running on campus， cinematic, anime aesthetic"





https://github.com/user-attachments/assets/7573165e-9665-405d-890a-8dd3da272815



<br/>

### Genshin Impact Character XiangLing LoRA Example (early tuned version)

1. **Download the Makima LoRA Model:**

   Download the Makima LoRA model from Huggingface:

   - [Xiangling Character LoRA](https://huggingface.co/svjack/Genshin_Impact_XiangLing_HunyuanVideo_lora_early)

   ```bash
   xiangling_test_epoch4.safetensors
   ```

   Copy the model to the `loras` directory:

   ```bash
   cp xiangling_test_epoch4.safetensors ComfyUI/models/loras
   ```

4. **Run the Workflow:**

   Create a Python script `run_t2v_xiangling_lora.py`:

   ```python
   #### character do something (seed 42)
   from comfy_script.runtime import *
   load()
   from comfy_script.runtime.nodes import *
   with Workflow():
       vae = HyVideoVAELoader(r'hunyuan_video_vae_bf16.safetensors', 'bf16', None)
       lora = HyVideoLoraSelect('xiangling_test_epoch4.safetensors', 2.0, None, None)
       model = HyVideoModelLoader(r'hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, None, lora)
       hyvid_text_encoder = DownloadAndLoadHyVideoTextEncoder('Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'openai/clip-vit-large-patch14', 'fp16', False, 2, 'disabled')
       hyvid_embeds = HyVideoTextEncode(hyvid_text_encoder, "solo,Xiangling, cook rice in a pot genshin impact ,1girl,highres,", 'bad quality video', 'video', None, None, None)
       samples = HyVideoSampler(model, hyvid_embeds, 478, 512, 85, 30, 6, 9, 42, 1, None, 1, None)
       images = HyVideoDecode(vae, samples, True, 64, 256, True)
       #_ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None)
       _ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None,
                           pix_fmt = 'yuv420p', crf=19, save_metadata = True, trim_to_audio = False)
   ```

   Run the script:

   ```bash
   python run_t2v_xiangling_lora.py
   ```

<br/>

- prompt = "solo,Xiangling, cook rice in a pot genshin impact ,1girl,highres,"



https://github.com/user-attachments/assets/f09a7bfc-08d2-41ea-86a0-85e5d048e4fe



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
