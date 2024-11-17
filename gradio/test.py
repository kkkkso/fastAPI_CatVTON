import argparse
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'
from datetime import datetime

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
torch.jit.script = lambda f: f
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# 패키지 추가
import cv2
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
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

args = parse_args()
'''
RESUME_PATH = os.getenv("RESUME_PATH", "zhengchong/CatVTON")
repo_path = snapshot_download(repo_id=RESUME_PATH)

# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

person_image = Image.open("./resource/demo/example/person/men/model_7.png").convert("RGB")
mask = automasker(
            person_image,
            'upper'
        )['mask'] # 여기서 리턴되는 mask는 PIL 이미지임.(cloth_masker.py 참조) # 참고로 ['densepose']로 densepose도 확인가능. 



### 여기서 mask modify에 사용된 코드를 app.py에 체크해놓은 부분에 추가하면 된다!
def remove_bottom_part(mask: np.ndarray, y_threshold: int):
    """
    이미지의 y_threshold 아래에 있는 부분을 삭제.
    :param mask: 입력 마스크 (numpy 배열)
    :param y_threshold: 제거할 Y 좌표 값
    :return: 수정된 마스크 (numpy 배열)
    """
    # y_threshold 아래의 모든 픽셀을 0으로 설정
    mask[y_threshold:, :] = 0
    return Image.fromarray(mask)


# closing 연산 / fitting_mode가 standard 나 loose 일때만 사용하기
def morph_close(mask):
    mask_np = np.array(mask)
    kernel = np.ones((30, 30), np.uint8) # 커질수록 잘 연결됨

    closed_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(closed_mask)


# opening 연산 / fitting_mode가 standard 나 loose 일때만 사용하기
def morph_open(mask):
    mask_np = np.array(mask)
    kernel = np.ones((30, 30), np.uint8) # 커질수록 잘 사라짐

    #closed_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel) #opened_mask는 numpy 연산 결과 이므로 PIL 이미지로 변환 필요
    
    return Image.fromarray(opened_mask)

# def morph_open2(mask):
    mask_np = np.array(mask)
    kernel = np.ones((150, 150), np.uint8) # 커질수록 잘 사라짐

    #closed_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel) #opened_mask는 numpy 연산 결과 이므로 PIL 이미지로 변환 필요
    
    return Image.fromarray(opened_mask)

## opened_mask = morph_open(mask)
## opened_mask.save('./opened_mask.png') #opened_mask는 PIL 이미지 형태로 반환되었으므로 (Image.fromarray()사용해서) .save를 바로 사용할 수 있다.

#opened_mask2 = morph_open2(mask)
#kernel = np.ones((50, 50), np.uint8)
#opened_mask2 = cv2.dilate(np.array(opened_mask2), kernel, iterations=1)
#opened_mask2 = Image.fromarray(opened_mask2)
#opened_mask2 = mask_processor.blur(opened_mask2, blur_factor=9)
#opened_mask2.save('./opened_mask2.png')



# mask = mask_processor.blur(mask, blur_factor=9)
## mask.save("./test_mask.png")  # 마스크를 PNG 파일로 저장
## masked_person = vis_mask(person_image, mask) # app.py에서도 blur 처리 한 다음에 vis_mask 메서드 호출함.
## masked_person.save("./test_masked_person.png")  # 마스크와 target img가 합쳐진 사진을 PNG 파일로 저장



# mask의 y축 음의 방향 이동
def extend_mask_downward(mask_image: np.ndarray, pixels: int) -> np.ndarray:
    """
    y축 음의 방향으로 (아래로) 마스크 이미지를 확장하는 함수.
    
    :param mask_image: 마스크 이미지 (numpy 배열)
    :param pixels: 확장할 픽셀 수
    :return: 확장된 마스크 이미지 (numpy 배열)
    """
    # 이진화된 마스크를 만듦
    mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)[1]

    # 확장을 위한 커널. y축으로만 확장하기 위해 세로 길이를 크게 설정함
    kernel = np.zeros((pixels, 1), np.uint8)  # y축으로만 길어진 커널

    # y축 음의 방향으로만 확장 (cv2.dilate 사용)
    extended_mask = cv2.dilate(mask, kernel, iterations=1)
    
    return Image.fromarray(extended_mask)


def image_equal(img1, img2):
    return np.array_equal(np.array(img1), np.array(img2))

# 마스크를 y축 음의 방향으로 50픽셀 확장
## extended_mask = extend_mask_downward(np.array(mask), pixels=100)

# 확장된 마스크 저장
## extended_mask.save('extended_mask_image.png')

# 최종 마스크 저장
# fitting 정도에 따라, extended_mask 함수 호출 변수인 pixels를 조절하면 된다.
# 정확도를 위해 그냥 dilation 하지 않고, y좌표가 약간 다른 두 마스크를 합쳤다.
## final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
## final_mask = morph_close(morph_open(final_mask)) #불필요한 동떨어진 부분 삭제 -> 연결되지 않은 부분 연결

## final_mask.save('final_mask_image.png')
## masked_person2 = vis_mask(person_image, final_mask) # app.py에서도 blur 처리 한 다음에 vis_mask 메서드 호출함.
## masked_person2.save("./test_masked_person2.png")  # 마스크와 target img가 합쳐진 사진을 PNG 파일로 저장




#person_image = Image.open("path_to_image").convert("RGB")
#standard_image = Image.open("./resource/demo/example/person/men/m_lvl3.png").convert("RGB")



"""
compare_image_mlvl3 = Image.open("./resource/demo/example/person/men/m_lvl3.png").convert("RGB")
compare_image_mlvl3 = resize_and_crop(compare_image_mlvl3, (args.width, args.height))

person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB") # 이걸 어느 bmi 레벨을 기준으로 쓸지는 뭐.. 실험해보면서 제일 좋은 거 정하면 됨.
person_image2 = resize_and_crop(person_image2, (args.width, args.height))
mask = automasker(
    person_image2,
    "upper"
)['mask']
mask.save("./first_mask.png")

# 이후 처리
sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
sam_mask_lower = resize_and_crop(sam_mask_lower, (args.width, args.height))
sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl3_upper_sam.png").convert("L")
sam_mask_upper = resize_and_crop(sam_mask_upper, (args.width, args.height))

mask_np = np.array(mask) 
sam_mask_upper_np = np.array(sam_mask_upper)
sam_mask_lower_np = np.array(sam_mask_lower)

kernel = np.ones((10, 10), np.uint8)
sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
result_np = np.where(sam_mask_upper_np== 255, 255, result_np)

mask = Image.fromarray(result_np)
mask.save("./last_mask2.png")
"""