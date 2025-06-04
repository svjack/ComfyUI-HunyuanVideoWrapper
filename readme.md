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
   #wget https://files.pythonhosted.org/packages/30/e8/a390dd2e83f468327b944bacc5cd2e787e0151f690fec9682a78130a488f/comfyui_frontend_package-1.21.6-py3-none-any.whl
   wget https://files.pythonhosted.org/packages/6f/41/23e60b0dac42da9a6a264a1a9a82046283aeddbe522717c14be4e85421fd/comfyui_frontend_package-1.21.7-py3-none-any.whl
   pip install comfyui_frontend_package-1.21.7-py3-none-any.whl
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
huggingface-cli download --resume-download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors --local-dir ComfyUI/models/diffusion_models --local-dir-use-symlinks False

huggingface-cli download --resume-download Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors --local-dir ComfyUI/models/vae --local-dir-use-symlinks False

huggingface-cli download --resume-download Kijai/HunyuanVideo_comfy hunyuan_video_custom_720p_fp8_scaled.safetensors --local-dir ComfyUI/models/diffusion_models --local-dir-use-symlinks False


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

```python 
###### python -m comfy_script.transpile hyvideo_custom_testing_01_edit.json

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

image_path = "爱可菲.webp"
prompt = 'Realistic, High-quality. A woman is boxing with a panda, and they are at a stalemate.'

with Workflow():
    image, _ = LoadImage(image_path)
    image, width, height = ImageResizeKJv2(image, 896, 512, 'lanczos', 'pad', '255,255,255', 'center', 16)
    PreviewImage(image)
    # _ = HyVideoTeaCache(0.10000000000000002, 'offload_device', 0, -1)
    vae = HyVideoVAELoader('hunyuan_video_vae_bf16.safetensors', 'bf16', None)
    torch_compile_args = HyVideoTorchCompileSettings('inductor', False, 'default', False, 64, True, True, False, False, False)
    block_swap_args = HyVideoBlockSwap(20, 0, False, False)
    model = HyVideoModelLoader('hunyuan_video_custom_720p_fp8_scaled.safetensors', 'bf16', 'fp8_scaled', 'offload_device', 'sageattn', torch_compile_args, block_swap_args, None, False, True)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp8_scaled.safetensors', 'hunyuan_video', 'default')
    clip_vision = CLIPVisionLoader('llava_llama3_vision.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'center')
    conditioning = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, prompt, 2)
    conditioning2 = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, 'Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.', 2)
    hyvid_embeds = HyVideoTextEmbedBridge(conditioning, 7.500000000000002, 0, 1, False, True, conditioning2)
    samples = HyVideoEncode(vae, image, False, 64, 256, True, 0, 1, 'sample')
    samples = HyVideoSampler(model, hyvid_embeds, width, height, 85, 30, 0, 13.000000000000002, 2, True, None, samples, 1, None, None, None, None, 'FlowMatchDiscreteScheduler', 0, 'dynamic', None, None, None, None)
    images = HyVideoDecode(vae, samples, True, 64, 256, True, 0, False)
    images2 = ImageConcatMulti(2, images, image, 'left', False)
    _ = VHSVideoCombine(images2, 24, 0, 'HunyuanVideoCustom_wrapper', 'video/h264-mp4', False, False, None, None, None)

from datasets import load_dataset
ds = load_dataset("svjack/daily-actions-locations-en-zh")
df = ds["train"].to_pandas()
df.to_csv("en_action.csv", index = False)

vim run_akf.py

import os
import time
import pandas as pd
import subprocess
from pathlib import Path
from itertools import product

# Configuration
SEEDS = [42]
IMAGE_PATHS = ['npl.jpg']  # Using the new image path
OUTPUT_DIR = 'ComfyUI/temp'
CSV_PATH = 'en_action.csv'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 300  # Increased timeout for video generation (5 minutes)
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(image_path, seed, action):
    """Generate the Hunyuan Video script with the given parameters"""
    prompt = f'Realistic, High-quality. the man {action}'

    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

image_path = "{image_path}"
prompt = '{prompt}'

with Workflow():
    image, _ = LoadImage(image_path)
    image, width, height = ImageResizeKJv2(image, 896, 512, 'lanczos', 'pad', '255,255,255', 'center', 16)
    PreviewImage(image)
    # _ = HyVideoTeaCache(0.10000000000000002, 'offload_device', 0, -1)
    vae = HyVideoVAELoader('hunyuan_video_vae_bf16.safetensors', 'bf16', None)
    torch_compile_args = HyVideoTorchCompileSettings('inductor', False, 'default', False, 64, True, True, False, False, False)
    block_swap_args = HyVideoBlockSwap(20, 0, False, False)
    model = HyVideoModelLoader('hunyuan_video_custom_720p_fp8_scaled.safetensors', 'bf16', 'fp8_scaled', 'offload_device', 'sageattn', torch_compile_args, block_swap_args, None, False, True)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp8_scaled.safetensors', 'hunyuan_video', 'default')
    clip_vision = CLIPVisionLoader('llava_llama3_vision.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'center')
    conditioning = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, prompt, 2)
    conditioning2 = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, 'Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.', 2)
    hyvid_embeds = HyVideoTextEmbedBridge(conditioning, 7.500000000000002, 0, 1, False, True, conditioning2)
    samples = HyVideoEncode(vae, image, False, 64, 256, True, 0, 1, 'sample')
    samples = HyVideoSampler(model, hyvid_embeds, width, height, 85, 30, 0, 13.000000000000002, 2, True, None, samples, 1, None, None, None, None, 'FlowMatchDiscreteScheduler', 0, 'dynamic', None, None, None, None)
    images = HyVideoDecode(vae, samples, True, 64, 256, True, 0, False)
    images2 = ImageConcatMulti(2, images, image, 'left', False)
    _ = VHSVideoCombine(images2, 24, 0, 'HunyuanVideoCustom_wrapper', 'video/h264-mp4', False, False, None, None, None)
"""
    return script_content

def main():
    # Load actions from CSV
    try:
        actions = pd.read_csv(CSV_PATH)["en_action"].tolist()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate all combinations of seeds and image paths
    seed_image_combinations = list(product(SEEDS, IMAGE_PATHS))

    # Main generation loop
    for action in actions:
        for seed, image_path in seed_image_combinations:
            # Generate script
            script = generate_script(image_path, seed, action)

            # Write script to file
            with open('run_hunyuan_video.py', 'w') as f:
                f.write(script)

            # Get current output count before running
            initial_count = get_latest_output_count()

            # Run the script
            print(f"Generating video with action: {action}, seed: {seed}, image: {image_path}")
            subprocess.run([PYTHON_PATH, 'run_hunyuan_video.py'])

            # Wait for new output
            if not wait_for_new_output(initial_count):
                print("Timeout waiting for new output. Continuing to next generation.")
                continue

if __name__ == "__main__":
    main()
```


https://github.com/user-attachments/assets/391fa7c8-0c88-45a2-9c2c-fde1b4facadb


```python
git clone https://huggingface.co/datasets/svjack/Xiang_InfiniteYou_Handsome_Pics_Captioned

import os
import time
import pandas as pd
import subprocess
from pathlib import Path
from itertools import product
from datasets import load_dataset
from PIL import Image

# Configuration
SEEDS = [42]
OUTPUT_DIR = 'ComfyUI/temp'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/bin/python'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 3000  # Increased timeout for video generation (5 minutes)
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(image_path, seed, prompt):
    """Generate the Hunyuan Video script with the given parameters"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

image_path = "{image_path}"
prompt = '{prompt}'

with Workflow():
    image, _ = LoadImage(image_path)
    image, width, height = ImageResizeKJv2(image, 896, 512, 'lanczos', 'pad', '255,255,255', 'center', 16)
    PreviewImage(image)
    # _ = HyVideoTeaCache(0.10000000000000002, 'offload_device', 0, -1)
    vae = HyVideoVAELoader('hunyuan_video_vae_bf16.safetensors', 'bf16', None)
    torch_compile_args = HyVideoTorchCompileSettings('inductor', False, 'default', False, 64, True, True, False, False, False)
    block_swap_args = HyVideoBlockSwap(20, 0, False, False)
    model = HyVideoModelLoader('hunyuan_video_custom_720p_fp8_scaled.safetensors', 'bf16', 'fp8_scaled', 'offload_device', 'sageattn', torch_compile_args, block_swap_args, None, False, True)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp8_scaled.safetensors', 'hunyuan_video', 'default')
    clip_vision = CLIPVisionLoader('llava_llama3_vision.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'center')
    conditioning = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, prompt, 2)
    conditioning2 = TextEncodeHunyuanVideoImageToVideo(clip, clip_vision_output, 'Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.', 2)
    hyvid_embeds = HyVideoTextEmbedBridge(conditioning, 7.500000000000002, 0, 1, False, True, conditioning2)
    samples = HyVideoEncode(vae, image, False, 64, 256, True, 0, 1, 'sample')
    samples = HyVideoSampler(model, hyvid_embeds, width, height, 85, 30, 0, 13.000000000000002, 2, True, None, samples, 1, None, None, None, None, 'FlowMatchDiscreteScheduler', 0, 'dynamic', None, None, None, None)
    images = HyVideoDecode(vae, samples, True, 64, 256, True, 0, False)
    images2 = ImageConcatMulti(2, images, image, 'left', False)
    _ = VHSVideoCombine(images2, 24, 0, 'HunyuanVideoCustom_wrapper', 'video/h264-mp4', False, False, None, None, None)
"""
    return script_content

def save_image(image_data, index):
    """Save image from dataset to file with zero-padded index"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    filename = f"{index:04d}.png"  # 4-digit zero-padded number
    filepath = os.path.join(INPUT_DIR, filename)

    if isinstance(image_data, Image.Image):
        image_data.save(filepath)
    else:
        # Handle case where image_data might be a dictionary or array
        Image.fromarray(image_data).save(filepath)

    return filepath

def main():
    # Load dataset
    try:
        ds = load_dataset("Xiang_InfiniteYou_Handsome_Pics_Captioned", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Main generation loop
    for i in range(len(ds)):
        # Get image and caption
        image_data = ds[i]["image"]
        prompt = ds[i]["joy-caption"].replace("'", "").replace('"', '')

        # Save image and get path
        image_path = save_image(image_data, i)
        image_path = image_path.split("/")[-1]

        for seed in SEEDS:
            # Generate script
            script = generate_script(image_path, seed, prompt)

            # Write script to file
            with open('run_hunyuan_video.py', 'w') as f:
                f.write(script)

            # Get current output count before running
            initial_count = get_latest_output_count()

            # Run the script
            print(f"Generating video for sample {i} with prompt: {prompt}, seed: {seed}")
            subprocess.run([PYTHON_PATH, 'run_hunyuan_video.py'])

            # Wait for new output
            if not wait_for_new_output(initial_count):
                print("Timeout waiting for new output. Continuing to next generation.")
                continue

if __name__ == "__main__":
    main()

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
