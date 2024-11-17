"""
api.py는 fastapi를 사용하고 있지만,
gradio를 사용한 웹 데모를 확인하고 싶다면 gradio 폴더 안에 있는 app-final.py를 사용하면 된다

api.py의 마스킹 부분을 제대로 수정한게,
api2.py임


"""


import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'
from datetime import datetime

from pydantic import BaseModel
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

from fastapi import FastAPI, File, Form, UploadFile
from typing import List
from typing import Optional
import shutil

from fastapi.responses import JSONResponse
import uuid
import base64
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print ('starting app')
# api 연결하면서 추가한 코드 
def pil_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # PNG 형식으로 저장
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# GPU에서 현재 할당된 메모리 확인 (GPU 0번 기준)
#allocated_memory = torch.cuda.memory_allocated(0) 
#print(f"GPU 0에서 할당된 메모리: {allocated_memory / (1024 ** 2)} MB")  # MB로 변환하여 출력


# 설정값을 환경 변수로 정의
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "booksforcharlie/stable-diffusion-inpainting")
RESUME_PATH = os.getenv("RESUME_PATH", "zhengchong/CatVTON")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "resource/demo/output")
WIDTH = int(os.getenv("WIDTH", 768))
HEIGHT = int(os.getenv("HEIGHT", 1024))

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


repo_path = snapshot_download(repo_id=RESUME_PATH)
print ('repo_path')
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL_PATH,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("no"),
    use_tf32=True,
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
    #person_image, mask = person_image["background"], person_image["layers"][0] # person_image["layers"][0]이 유저가 그린 마스크 레이어임.
    #mask = Image.open(mask).convert("L")
    #if len(np.unique(np.array(mask))) == 1:
    #    mask = None # 사용자가 마스크를 그리지 않은 경우.
    #else:
    #    mask = np.array(mask)
    #    mask[mask > 0] = 255 # 배경이 검은색.
    #    mask = Image.fromarray(mask)
    mask = None
    tmp_folder = "resource/demo/output"
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (768, 1024))
    cloth_image = resize_and_padding(cloth_image, (768, 1024))
    
    #예외처리
    #man
    compare_image_mlvl0 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
    compare_image_mlvl0 = resize_and_crop(compare_image_mlvl0, (768, 1024))

    compare_image_mlvl1 = Image.open("./resource/demo/example/person/men/m_lvl1.png").convert("RGB")
    compare_image_mlvl1 = resize_and_crop(compare_image_mlvl1, (768, 1024))

    compare_image_mlvl2 = Image.open("./resource/demo/example/person/men/m_lvl2.png").convert("RGB")
    compare_image_mlvl2 = resize_and_crop(compare_image_mlvl2, (768, 1024))

    compare_image_mlvl3 = Image.open("./resource/demo/example/person/men/m_lvl3.png").convert("RGB")
    compare_image_mlvl3 = resize_and_crop(compare_image_mlvl3, (768, 1024))

    #womam
    compare_image_wlvl0 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
    compare_image_wlvl0 = resize_and_crop(compare_image_wlvl0, (768, 1024))
    
    compare_image_wlvl1 = Image.open("./resource/demo/example/person/women/w_lvl1.png").convert("RGB")
    compare_image_wlvl1 = resize_and_crop(compare_image_wlvl1, (768, 1024))
    
    compare_image_wlvl2 = Image.open("./resource/demo/example/person/women/w_lvl2.png").convert("RGB")
    compare_image_wlvl2 = resize_and_crop(compare_image_wlvl2, (768, 1024))
    
    compare_image_wlvl3 = Image.open("./resource/demo/example/person/women/w_lvl3.png").convert("RGB")
    compare_image_wlvl3 = resize_and_crop(compare_image_wlvl3, (768, 1024))

    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (768, 1024))
    else:
        if image_equal(person_image, compare_image_mlvl3):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)

        elif image_equal(person_image, compare_image_wlvl3):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl3_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
            sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)
            

        elif image_equal(person_image, compare_image_mlvl2):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
            sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)
            


        elif image_equal(person_image, compare_image_wlvl2):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl2_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
            sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)



        elif image_equal(person_image, compare_image_mlvl1):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
            sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)

        elif image_equal(person_image, compare_image_wlvl1):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl1_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)
            sam_mask_lower_np = cv2.dilate(sam_mask_lower_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)

        elif image_equal(person_image, compare_image_mlvl0):
            person_image2 = Image.open("./resource/demo/example/person/men/m_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/m_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)              

        elif image_equal(person_image, compare_image_wlvl0):
            person_image2 = Image.open("./resource/demo/example/person/women/w_lvl0.png").convert("RGB")
            person_image2 = resize_and_crop(person_image2, (768, 1024))
            mask = automasker(
                person_image2,
                cloth_type
            )['mask']
            # 이후 처리
            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
            sam_mask_upper = Image.open("./resource/demo/example/person/sam/w_lvl0_upper_sam.png").convert("L")
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))

            mask_np = np.array(mask) 
            sam_mask_upper_np = np.array(sam_mask_upper)
            sam_mask_lower_np = np.array(sam_mask_lower)

            kernel = np.ones((10, 10), np.uint8)
            sam_mask_upper_np = cv2.dilate(sam_mask_upper_np, kernel, iterations=1)

            result_np = np.where(sam_mask_lower_np== 255, 0, mask_np)
            result_np = np.where(sam_mask_upper_np== 255, 255, result_np)
            mask = Image.fromarray(result_np)

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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=100)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_mlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_mlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/m_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl3) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl3_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl2) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl2_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask
            
        elif image_equal(person_image, compare_image_wlvl1) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl1_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
            sam_mask_upper = resize_and_crop(sam_mask_upper, (768, 1024))
            sam_mask_upper_np = np.array(sam_mask_upper)

            extended_mask = extend_mask_downward(sam_mask_upper_np, pixels=200)

            #최종 마스크 처리 (test.py 설명 참고)
            final_mask = Image.fromarray(np.array(opened_mask) | np.array(extended_mask))
            final_mask = morph_close(morph_open(final_mask))
            mask = final_mask

        elif image_equal(person_image, compare_image_wlvl0) and cloth_type == "lower":
            opened_mask = morph_open(mask)

            sam_mask_lower = Image.open("./resource/demo/example/person/sam/w_lvl0_lower_sam.png").convert("L")
            sam_mask_lower = resize_and_crop(sam_mask_lower, (768, 1024))
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
        return {"result_image": result_image, "masked_person": masked_person}
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


###### Fastapi의 api




# FastAPI 함수 정의
@app.post("/process-image")
async def process_image(
    cloth_type: str = Form(...),
    fitting_type: str = Form(...),
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    try:
        # 고유한 파일 이름 생성
        person_filename = f"received_{uuid.uuid4().hex}_{person_image.filename}"
        cloth_filename = f"received_{uuid.uuid4().hex}_{cloth_image.filename}"

        print ('person_filename: ', person_filename)
        print ('cloth_filename: ', cloth_filename)

        # 이미지 저장 디렉토리 생성
        os.makedirs("uploads", exist_ok=True)

        # 업로드된 이미지 저장
        person_path = os.path.join("uploads", person_filename)
        cloth_path = os.path.join("uploads", cloth_filename)

        with open(person_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)

        with open(cloth_path, "wb") as buffer:
            shutil.copyfileobj(cloth_image.file, buffer)

        # 이미지 처리 함수 호출
        result = submit_function(
            person_image=person_path,
            cloth_image=cloth_path,
            cloth_type=cloth_type,
            fitting_type=fitting_type,
            num_inference_steps=25,  
            guidance_scale=2.5,       
            seed=42,                  
            show_type='result only'   
        )

        print ('processing done')
        # 반환된 이미지 추출
        result_image = result['result_image']
        masked_person = result['masked_person']
        result_image.save('results/result.png')
        # 이미지를 Base64로 인코딩
        result_image_b64 = pil_to_base64(result_image)
        masked_person_b64 = pil_to_base64(masked_person)

        # 임시 파일 삭제 (필요 시)
        os.remove(person_path)
        os.remove(cloth_path)

        return {
            "message": "이미지가 처리되었습니다",
            "result_image": result_image_b64,
            "masked_person": masked_person_b64
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"오류 발생: {str(e)}"})




@app.post("/send-to-ssh")
async def send_to_ssh(
    cloth_type: str = Form(...),
    fitting_type: str = Form(...),
    person_image: UploadFile = File(...),  # 이미지 파일 업로드로 처리
    cloth_image: UploadFile = File(...)
):
    # 받은 데이터를 처리하거나 저장하는 로직
    return {"message": "데이터가 성공적으로 처리되었습니다."}

@app.get('/test')
async def test():
    return JSONResponse(status_code=200, content={"message": "hello"})