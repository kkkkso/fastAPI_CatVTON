import argparse
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'
from datetime import datetime

import gradio as gr
import spaces
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
torch.jit.script = lambda f: f
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from test import morph_close, morph_open, extend_mask_downward, image_equal
import cv2

# GPU에서 현재 할당된 메모리 확인 (GPU 0번 기준)
#allocated_memory = torch.cuda.memory_allocated(0)  # 0번 GPU에서 할당된 메모리 양을 반환
#print(f"GPU 0에서 할당된 메모리: {allocated_memory / (1024 ** 2)} MB")  # MB로 변환하여 출력


# to chck
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        # default="Kwai-Kolors/Kolors-Inpainting",
        default="booksforcharlie/stable-diffusion-inpainting",
        
        # default="stabilityai/stable-diffusion-2-inpainting",
        # default="runwayml/stable-diffusion-inpainting",
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

@spaces.GPU(duration=120)
# 매개변수로 fitting_type 추가해야 함. cloth_type 밑에.
def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    fitting_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    person_image, mask = person_image["background"], person_image["layers"][0] # person_image["layers"][0]이 유저가 그린 마스크 레이어임.
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None # 사용자가 마스크를 그리지 않은 경우.
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255 # 배경이 검은색.
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    #예외처리
    #man
    compare_image_mlvl0 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
    compare_image_mlvl0 = resize_and_crop(compare_image_mlvl0, (args.width, args.height))

    compare_image_mlvl1 = Image.open("./resource/demo/example/person/men/m_lvl1.png").convert("RGB")
    compare_image_mlvl1 = resize_and_crop(compare_image_mlvl1, (args.width, args.height))

    compare_image_mlvl2 = Image.open("./resource/demo/example/person/men/m_lvl2.png").convert("RGB")
    compare_image_mlvl2 = resize_and_crop(compare_image_mlvl2, (args.width, args.height))

    compare_image_mlvl3 = Image.open("./resource/demo/example/person/men/m_lvl3.png").convert("RGB")
    compare_image_mlvl3 = resize_and_crop(compare_image_mlvl3, (args.width, args.height))

    #womam
    compare_image_wlvl0 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
    compare_image_wlvl0 = resize_and_crop(compare_image_wlvl0, (args.width, args.height))
    
    compare_image_wlvl1 = Image.open("./resource/demo/example/person/women/w_lvl1.png").convert("RGB")
    compare_image_wlvl1 = resize_and_crop(compare_image_wlvl1, (args.width, args.height))
    
    compare_image_wlvl2 = Image.open("./resource/demo/example/person/women/w_lvl2.png").convert("RGB")
    compare_image_wlvl2 = resize_and_crop(compare_image_wlvl2, (args.width, args.height))
    
    compare_image_wlvl3 = Image.open("./resource/demo/example/person/women/w_lvl3.png").convert("RGB")
    compare_image_wlvl3 = resize_and_crop(compare_image_wlvl3, (args.width, args.height))

    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        if image_equal(person_image, compare_image_mlvl3):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam_v2.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)
            
            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            

        elif image_equal(person_image, compare_image_wlvl3):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam_v2.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            

        elif image_equal(person_image, compare_image_mlvl2):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam_v2.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            


        elif image_equal(person_image, compare_image_wlvl2):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam_v2.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            



        elif image_equal(person_image, compare_image_mlvl1):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            

        elif image_equal(person_image, compare_image_wlvl1):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            

        elif image_equal(person_image, compare_image_mlvl0):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
                         

        elif image_equal(person_image, compare_image_wlvl0):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (args.width, args.height))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            if cloth_type == "upper":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
                result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            elif cloth_type == "lower":
                kernel = np.ones((10, 10), np.uint8)
                sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)
                result_np = np.where(sam_mask_upper_np== 255, 0, mask_np)
                result_np = np.where(sam_mask_lower_np== 255, 255, result_np)
                mask = Image.fromarray(result_np)
            else:
                mask = Image.fromarray(mask_np)
            

        else:
            mask = automasker(
                person_image,
                cloth_type
            )['mask']

    # mask.save("./app_mask_created.png")

        # 가끔 bmi지수 높은 아바타의 경우, upper mask를 정확히 생성해내지 못하는 경우가 있어 수동으로 한 번 더 처리해줌.
        
        # 튀어나온 부분 밀어버리기 (는 사용자가 그린 mask에 대해서는 시행되면 안되므로, else문 안에 넣어두기)
        #if cloth_type == "upper":
        #    height = (np.array(mask)).shape[0]
        #    y_threshold = int(height * 0.7) # 이미지 높이의 50퍼센트 이하. 50퍼센트가 딱 적당함.

            # 밑부분 제거
        #    mask = remove_bottom_part(np.array(mask), y_threshold)
        # 위 방법으로 해결 불가임. 튀어나온 부분
        # input 된 target 이미지마다, 생성되는 mask 영역의 크기가 다르기 때문. mask 파일 자체의 크기는 같을 지언정.
        



    # 추가로 Fitting Type에 따라 마스크 처리 (else문 내부)
    if fitting_type == "standard":

        # mlvl3에 대한 upper lower 각각.
        if image_equal(person_image, compare_image_mlvl3) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # mlvl2에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # mlvl1에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
        
        # mlvl0에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        
        # wlvl3에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # wlvl2에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # wlvl1에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
        
        # wlvl0에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # 그 외 디폴트
        else:
            opened_mask = morph_open(mask)
            extended_mask = extend_mask_downward(np.array(mask), pixels=100)
            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

    elif fitting_type == "loose" :
        # mlvl3에 대한 upper lower 각각.
        if image_equal(person_image, compare_image_mlvl3) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # mlvl2에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # mlvl1에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
        
        # mlvl0에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        
        # wlvl3에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # wlvl2에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # wlvl1에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
        
        # wlvl0에 대한 upper lower 각각.
        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "upper":
            opened_mask = morph_open(mask)

            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
            sam_mask_lower_np = np.array(sam_mask_lower)

            extended_mask = extend_mask_downward(sam_mask_lower_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        # 그 외 디폴트
        else:
            opened_mask = morph_open(mask)
            extended_mask = extend_mask_downward(np.array(mask), pixels=200)
            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
        

    # 블러처리        
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image


def person_example_fn(image_path):
    return image_path


HEADER = """

"""

def app_gradio():
    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image = gr.ImageEditor(
                        interactive=True, label="Person Image", type="filepath"
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Fitting Type` to generate automatically </span>'
                        )
                        fitting_type = gr.Radio(
                            label="Try-On Fitting Type",
                            choices=["fit", "standard", "loose"],
                            value="fit", # default
                        )


                submit = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )
                
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    # Guidence Scale
                    guidance_scale = gr.Slider(
                        label="CFG Strenth", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                    )
                    # Random Seed
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "men", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ①",
                        )
                        women_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "women", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples come from the demos of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>. </span>'
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "upper", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "overall", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "person", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference Person Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "lower", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "lower"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference lower Examples",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>'
                        )

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            #여기도 매개변수 fitting_type 추가해야 함.
            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    fitting_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()