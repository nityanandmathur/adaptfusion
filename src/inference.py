import argparse
from typing import List
import torch
from PIL import Image
import os
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from tqdm import tqdm

MODEL_NAME = 'diffusers/sdxl-instructpix2pix-768'
PROMPT = 'make it sks scene'

def load_transform_images(source_path:str=None, width:int=1024, height:int=1024) -> Image.Image: # type: ignore
    """
    Load and transform image from the specified source path.
    Args:
        source_path (str, optional): The path to the directory containing the images. Defaults to None.
        width (int, optional): The target width of the images. Defaults to 1024.
        height (int, optional): The target height of the images. Defaults to 1024.
    Returns:
        Image.Image: Trnasformed Image.
    """
    image = Image.open(source_path)
    w, h = image.size
    #* Center crop to target resolution
    image = image.crop((w//2 - width//2, h//2 - height//2, w//2 + width//2, h//2 + height//2))
    return image

def load_pipeline(lora_path:str=None, layer:str='111', device:str='cuda:0') -> StableDiffusionXLInstructPix2PixPipeline: # type: ignore
    """
    Load the pipeline for StableDiffusionXLInstructPix2Pix model.
    Args:
        lora_path (str, optional): Path to the LORA weights file. Defaults to None.
        layer (str, optional): Layer configuration for adapter weight scales. Defaults to '111'.
    Returns:
        StableDiffusionXLInstructPix2PixPipeline: The loaded pipeline object.
    """

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(MODEL_NAME,
                                                                    torch_dtype=torch.float16).to(device)
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="adapter")
    adapter_weight_scales = { "unet": { "down": int(layer[0]), "mid": int(layer[1]), "up": int(layer[2])}}
    pipe.set_adapters("adapter", adapter_weight_scales)

    return pipe

def main():
    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--source', type=str, required=True, help='Path to the source image folder')
    parser.add_argument('--target', type=str, required=True, help='Path to the target image folder')
    parser.add_argument('--lora', type=str, required=True, help='Path to the LoRA checkpoint')
    parser.add_argument('--layer', type=str, default='111', help='Layer to be used for the fusion, format: DMU')
    parser.add_argument('--width', type=int, default=1600, help='Width of the output image')
    parser.add_argument('--height', type=int, default=900, help='Height of the output image')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    source_images = os.listdir(args.source)
    pipeline = load_pipeline(args.lora, args.layer, args.device)
    if not os.path.exists(args.target):
        os.makedirs(args.target, exist_ok=True)

    for path in tqdm(source_images):
        image = load_transform_images(os.path.join(args.source, path), args.width, args.height)
        output = pipeline(prompt=PROMPT, image=image, num_inference_steps=30,
                          width=args.width, height=args.height,
                          generator=torch.manual_seed(args.seed)).images[0]
        output.save(os.path.join(args.target, path))

if __name__ == '__main__':
    main()
