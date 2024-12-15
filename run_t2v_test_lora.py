from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    vae = HyVideoVAELoader(r'hunyuan_video_vae_bf16.safetensors', 'bf16', None)
    lora = HyVideoLoraSelect('hunyuan_video_test_epoch2.safetensors', 1.5, None, None)
    model = HyVideoModelLoader(r'hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, None, lora)
    hyvid_text_encoder = DownloadAndLoadHyVideoTextEncoder('Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'openai/clip-vit-large-patch14', 'fp16', False, 2, 'disabled')
    hyvid_embeds = HyVideoTextEncode(hyvid_text_encoder, "The video presents a tranquil scene of a small, isolated island with a rocky outcrop. The island is covered in lush greenery and dotted with vibrant pink cherry blossom trees in full bloom.", 'bad quality video', 'video', None, None, None)
    samples = HyVideoSampler(model, hyvid_embeds, 512, 320, 85, 30, 6, 9, 6, 1, None, 1, None)
    images = HyVideoDecode(vae, samples, True, 64, 256, True)
    #_ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None)
    _ = VHSVideoCombine(images, 24, 0, 'HunyuanVideo', 'video/h264-mp4', False, True, None, None, None,
                        pix_fmt = 'yuv420p', crf=19, save_metadata = True, trim_to_audio = False)
