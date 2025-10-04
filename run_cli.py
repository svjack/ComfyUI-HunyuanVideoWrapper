'''
#!/bin/bash
# run_batch_comfy.sh

# 设置Comfy环境路径（必须修改为您的实际路径）
COMFY_ENV="/environment/miniconda3/envs/system/bin/comfy"

# 设置工作流文件路径
WORKFLOW_FILE="wan_camera_api.json"

# 运行批量处理脚本
python -m run_cli \
    --dataset svjack/mimi_AI_art_Infinite_future_landscape_images_captioned \
    --workflow $WORKFLOW_FILE \
    --comfy-env $COMFY_ENV \
    --timeout 600 \
    --output-dir ComfyUI/output \
    --input-dir ComfyUI/input

'''
import os
import json
import subprocess
import tempfile
from datasets import load_dataset
from PIL import Image
import argparse

def process_images_with_comfyui(dataset_path, comfyui_workflow_path, comfy_env_path, timeout_seconds=600, output_dir="ComfyUI/output", input_dir="ComfyUI/input"):
    """
    批量处理huggingface数据集中的图像和提示词（逐个迭代版本）
    
    Args:
        dataset_path: huggingface数据集路径
        comfyui_workflow_path: ComfyUI工作流JSON文件路径
        comfy_env_path: Comfy环境路径（必须设置）
        timeout_seconds: 超时时间（秒），默认10分钟（600秒）
        output_dir: 输出目录
        input_dir: 输入目录
    """
    
    # 验证comfy环境路径是否存在[5](@ref)
    if not os.path.exists(comfy_env_path):
        raise FileNotFoundError(f"Comfy环境路径不存在: {comfy_env_path}")
    
    # 创建目录[3](@ref)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    # 确定是train还是validation split
    split = 'train' if 'train' in dataset else 'validation' if 'validation' in dataset else list(dataset.keys())[0]
    data = dataset[split]
    
    # 加载工作流模板
    with open(comfyui_workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # 逐个处理每个数据项[1,2](@ref)
    for idx, item in enumerate(data):
        # 保存图像到输入目录
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image_filename = f"image_{idx:06d}.png"
        image_path = os.path.join(input_dir, image_filename)
        image.save(image_path)
        
        # 获取提示词
        prompt_text = item['prompt']
        
        # 修改工作流
        modified_workflow = workflow.copy()
        
        # 更新图像路径
        modified_workflow["79"]["inputs"]["image"] = image_filename
        
        # 更新提示词
        modified_workflow["81"]["inputs"]["text"] = prompt_text
        
        # 创建临时工作流文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(modified_workflow, temp_file, indent=2)
            temp_workflow_path = temp_file.name
        
        try:
            # 构建命令 - 直接使用comfy环境路径[5](@ref)
            cmd = [comfy_env_path, "run", "--workflow", temp_workflow_path, "--wait", "--timeout", str(timeout_seconds)]
            
            # 运行命令[6,7,8](@ref)
            print(f"Processing image {idx}: {image_filename}")
            print(f"Prompt: {prompt_text[:100]}...")
            print(f"Timeout setting: {timeout_seconds} seconds")
            print(f"Executing command: {' '.join(cmd)}")
            
            # 执行命令并捕获输出，设置超时时间[6](@ref)
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout_seconds + 30  # 额外增加30秒缓冲
            )
            
            # 无论是否报错都打印输出
            print("=== Command Execution Result ===")
            print(f"Return code: {result.returncode}")
            print("--- STDOUT ---")
            print(result.stdout if result.stdout else "(No stdout output)")
            print("--- STDERR ---")
            print(result.stderr if result.stderr else "(No stderr output)")
            print("=================")
            
            if result.returncode == 0:
                print(f"✓ Successfully processed image {idx}")
            else:
                print(f"✗ Error processing image {idx}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout ({timeout_seconds} seconds) expired while processing image {idx}")
        except Exception as e:
            print(f"✗ Exception occurred while processing image {idx}: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_workflow_path):
                os.unlink(temp_workflow_path)
        
        print(f"Progress: {idx + 1}/{len(data)} ({(idx + 1)/len(data)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='批量处理huggingface数据集图像（逐个迭代版本）')
    parser.add_argument('--dataset', type=str, default='svjack/mimi_AI_art_Infinite_future_landscape_images_captioned',
                       help='huggingface数据集路径')
    parser.add_argument('--workflow', type=str, required=True,
                       help='ComfyUI工作流JSON文件路径')
    parser.add_argument('--comfy-env', type=str, required=True,
                       help='Comfy环境路径（必须设置）')
    parser.add_argument('--timeout', type=int, default=600,
                       help='超时时间（秒），默认10分钟（600秒）')
    parser.add_argument('--output-dir', type=str, default='ComfyUI/output',
                       help='输出目录')
    parser.add_argument('--input-dir', type=str, default='ComfyUI/input',
                       help='输入目录')
    
    args = parser.parse_args()
    
    # 验证超时时间合理性
    if args.timeout <= 0:
        raise ValueError("超时时间必须大于0秒")
    
    print(f"使用配置:")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 工作流: {args.workflow}")
    print(f"  - Comfy环境: {args.comfy_env}")
    print(f"  - 超时时间: {args.timeout}秒 ({args.timeout/60:.1f}分钟)")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 输入目录: {args.input_dir}")
    
    # 处理图像
    process_images_with_comfyui(
        dataset_path=args.dataset,
        comfyui_workflow_path=args.workflow,
        comfy_env_path=args.comfy_env,
        timeout_seconds=args.timeout,
        output_dir=args.output_dir,
        input_dir=args.input_dir
    )

if __name__ == "__main__":
    main()
