CUDA_VISIBLE_DEVICES=1
accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --instance_data_dir="content" \
 --output_dir="output/content" \
 --instance_prompt="A [v1]" \
 --resolution=256 \
 --rank=64 \
 --train_batch_size=1 \
 --learning_rate=5e-5 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=1000 \
 --checkpointing_steps=1100 \
 --seed="0" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision="fp16"


 accelerate launch train_dreambooth_b-lora_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --instance_data_dir="Styles/cartoon_line" \
 --output_dir="output/style" \
 --instance_prompt="A [v3]" \
 --resolution=256 \
 --rank=64 \
 --train_batch_size=1 \
 --learning_rate=5e-5 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=1000 \
 --checkpointing_steps=1100 \
 --seed="0" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision="fp16"


python inference.py --prompt="A <v1> in <v3> style" --content_B_LoRA="output/content" --style_B_LoRA="output/style" --output_path="output/results" --num_images_per_prompt=1

python inference.py --prompt="A <v18> in <v30> style" --content_B_LoRA='lora-library/B-LoRA-teddybear' --style_B_LoRA='lora-library/B-LoRA-pen_sketch' --output_path="output/results" --num_images_per_prompt=1