import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from blora_utils import BLOCKS, filter_lora, scale_lora
import matplotlib.pyplot as plt

def plot_images(images, titles=None, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)

    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.show()
def load_b_lora_to_unet(pipe, content_lora_model_id: str = '', style_lora_model_id: str = '', content_alpha: float = 1.,
                            style_alpha: float = 1.) -> None:
        try:
            # Get Content B-LoRA SD
            if content_lora_model_id:
                content_B_LoRA_sd, _ = pipe.lora_state_dict(content_lora_model_id)
                content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
                content_B_LoRA = scale_lora(content_B_LoRA, content_alpha)
            else:
                content_B_LoRA = {}

            # Get Style B-LoRA SD
            if style_lora_model_id:
                style_B_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
                style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
                style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
            else:
                style_B_LoRA = {}

            # Merge B-LoRAs SD
            res_lora = {**content_B_LoRA, **style_B_LoRA}

            # Load
            pipe.load_lora_into_unet(res_lora, None, pipe.unet)
        except Exception as e:
            raise type(e)(f'failed to load_b_lora_to_unet, due to: {e}')
        
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,local_files_only=True)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    # 图像风格化
    content_B_LoRA_path = 'lora-library/B-LoRA-teddybear' #缓存在/home/ludan/.cache/huggingface/hub文件下
    style_B_LoRA_path = 'lora-library/B-LoRA-pen_sketch'
    content_alpha,style_alpha = 1,1.1

    load_b_lora_to_unet(pipeline, content_B_LoRA_path, style_B_LoRA_path, content_alpha, style_alpha)

    prompt = 'A [v18] in [v30] style'
    image = pipeline(prompt,generator=torch.Generator(device="cuda").manual_seed(48), num_images_per_prompt=1).images[0].resize((512,512))
    # Save
    image.save(f'output.jpg')
