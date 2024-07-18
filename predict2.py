import re
import os
import sys
import shutil
import requests
import argparse
import random
import fnmatch
import traceback
import subprocess
import base64
import random

# 自動安裝需要的庫
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.run(["pip", "install", package], check=True)
        __import__(package)

# 嘗試導入所需的庫
libraries = [
    'imgutils',
    'tqdm',
    'PIL',
    'pathlib',
    'datetime',
    'torch',
    'onnxruntime',
    'aesthetic_predictor_v2_5',
    'moellava'
]

for lib in libraries:
    install_and_import(lib)

import torch
from io import BytesIO
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import onnxruntime
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

# 嘗試導入 dghs-imgutils[gpu]，如果未安裝，則跳過相關處理
try:
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
    from imgutils.validate import anime_dbrating
except ImportError:
    print("正在安裝 dghs-imgutils[gpu]...")
    subprocess.run(["pip", "install", "dghs-imgutils[gpu]"], check=True)
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
    from imgutils.validate import anime_dbrating


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disable_torch_init()
model_path = 'LanguageBind/MoE-LLaVA-StableLM-1.6B-4e-384' if args.low_vram else 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)

aes_model, aes_preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
aes_model = aes_model.to(torch.bfloat16).to(device)

def generate_response(tokenizer, model, image_processor, image, wd14_caption, chartags):

    image_tensor = processor['image'].preprocess(image.convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
    conv_mode = "phi"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    default_prompt = f"\nUse {chartags} as the name of person(s). What happens in this image? What does this image appears to be? How about this image? Give me detailed and long description with {wd14_caption}"
    prompt = DEFAULT_IMAGE_TOKEN + default_prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=200,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip().replace('\n', ' ')
    caption = outputs.replace('\n', ' ').replace(', ', ' ').replace('.', ',')
    other_tags = [tag for tag in chartags.split(', ') + wd14_caption.split(', ')  if tag not in caption]
    firsttag, caption = caption.split(', ', 1) if ', ' in caption else (caption, '')
    caption = ', '.join(other_tags) + ', ' + caption 
    
    if caption[-1]==',':
        return caption[:-1], firsttag
    else:        
        return caption, firsttag

def get_aesthetic_tag(image):
    def aesthetic_tag(score):
        if score >= 6:
            return "aesthetic."
        elif score >= 5:
            return "looks good."
        elif score >= 4.5:
            return "looks bad."
        else:
            return "looks very bad."
    pixel_values = (
        aes_preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .to(device)
    )
    with torch.inference_mode():
        score = aes_model(pixel_values).logits.squeeze().float().cpu().numpy()
        aestag = aesthetic_tag(score)
    return aestag

def generate_special_text(image_path, args, features=None, chars=None):
    """
    根據 features, image_path 和 parent_folder 生成 special_text。
    """
    def has_reverse_name(name_set, name):
        """
        檢查 name_set 中是否存在 name 的相反名稱（中間有一個空格）。
        """
        name_parts = name.split()
        if len(name_parts) == 2:
            reverse_name = f"{name_parts[1]} {name_parts[0]}"
            if reverse_name in name_set:
                return True
        return False
    base_file_name = os.path.splitext(image_path)[0]
    boorutag_path = None
    boorutag = ""
    artisttag = ""
    styletag = None
    chartag_from_folder = ""
    concept_tag = ""
    # 查找 boorutag 文件路徑
    for ext in ['.jpg.boorutag', '.png.boorutag']:
        potential_path = base_file_name + ext
        if os.path.exists(potential_path):
            boorutag_path = potential_path
            break

    chartags = set()

    # 獲取 parent_folder 並添加 chartag_from_folder
    parent_folder = Path(image_path).parent.name
    if args.folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        chartag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip().lower()
        chartags.add(chartag_from_folder)            
            
    # 處理 boorutag 文件內容
    if boorutag_path:
        try:
            with open(boorutag_path, 'r', encoding='cp950') as file:
                lines = file.readlines()
                first_line = lines[0]
                first_line_cleaned = re.sub(r'\(.*?\)', '', first_line)
                for tag in first_line_cleaned.split(','):
                    cleaned_tag = tag.replace('\\', '').replace('_', ' ').strip()
                    if not has_reverse_name(chartags, cleaned_tag):
                        chartags.add(cleaned_tag)
                if len(lines) >= 19:
                    artisttag = lines[6].strip()
                    boorutag = lines[18].strip()
                    boorutag_tags = drop_overlap_tags(boorutag.split(', '))
                    boorutag_tags_cleaned = [tag for tag in boorutag_tags if tag.replace(' ', '_') not in features.keys()]
                    boorutag = ', ' + ', '.join(boorutag_tags_cleaned)                
        except Exception as e:
            # 讀取文件或處理過程中發生錯誤
            pass

    # 處理 chars.keys()
    if chars:
        for key in chars.keys():
            cleaned_key = re.sub(r'\(.*?\)', '', key).replace('\\', '').replace('_', ' ').strip()
            if not has_reverse_name(chartags, cleaned_key):
                chartags.add(cleaned_key)

    # 將 chartags 轉換為列表並隨機打亂
    chartags = list(chartags)
    random.shuffle(chartags)

    if chartag_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{'focus on ' if 'solo_focus' in features else ''}one person {chartag_from_folder}", ', '.join(chartags), boorutag, artisttag

    if len(chartags) > 3:
        chartags = []
    
    if not chartag_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{'focus on ' if 'solo_focus' in features else ''}one person {' '.join(chartags)}" if chartags else "", ', '.join(chartags), boorutag, artisttag

    return f"{'include ' if chartags else ''}{' and '.join(chartags)}", ', '.join(chartags), boorutag, artisttag

def process_image(image_path, args):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """

    def resize_image(image_path, max_size=448):
        """
        縮小圖像使其最大邊不超過 max_size，返回縮小後的圖像數據
        """
        image = Image.open(image_path)
        if max(image.width, image.height) > max_size:
            if image.width > image.height:
                new_width = max_size
                new_height = int(max_size * image.height / image.width)
            else:
                new_height = max_size
                new_width = int(max_size * image.width / image.height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def process_features(features: dict) -> (dict, str):
        patterns_to_keep = [
            r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$', r'^sketch$', 
            r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', 
            r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
            r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^.*realistic$', 
            r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
            r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$'            
        ]
        keep_tags_set = set()
        #if 'solo' in features or 'solo_focus' in features:
        #    patterns_to_keep.extend([r'^holding_.*$'])
            #, r'^.*grab.*$', r'^.*lift.*$', r'^.*pull$', r'^.*_own_.*$', r'^.*covered.*$', r'^.*_masturbation.*$', r'^.*out.*$', r'^.*_between_.*$'
        keys = list(features.keys())
        keys_to_delete = []
        
        for key in keys:
            for pattern in patterns_to_keep:
                regex = re.compile(pattern)
                if regex.match(key):
                    keep_tags_set.add(key.replace('_', ' '))
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            if key in features:
                del features[key]
        
        keep_tags = ', '.join(keep_tags_set).rstrip(', ')
        
        return features, keep_tags

    def format_wd14_caption(wd14_caption):       
        tags = wd14_caption.split(", ")
        tags_to_delete = []
        lying_conditions = ['on stomach', 'on back', 'on side']
        if 'lying' in tags and any(cond in tags for cond in lying_conditions):
            for cond in lying_conditions:
                if cond in tags:
                    tags.append(f'lying {cond}')
                    tags_to_delete.append(cond)
            tags_to_delete.append('lying')
            
        boygirl_tags = [tag for tag in tags if tag in {'multiple girls', '1girl', 'multiple boys', '1boy'}]
        if boygirl_tags:
            boygirl_tag = ' '.join(sorted(boygirl_tags))
            tags.append(boygirl_tag)
            for tag in boygirl_tags:
                tags_to_delete.append(tag)         
        tags = [tag for tag in tags if tag not in tags_to_delete]
        wd14_caption = ', '.join(tags)
        return wd14_caption

    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

    try:
        image = resize_image(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 使用 imgutils 獲取圖片等級
        rating, features, chars = get_wd14_tags(image, character_threshold=0.6, general_threshold=0.3, drop_overlap=True)
        features, keeptag = process_features(features)
        wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
        special_text, chartags, boorutag, artisttag = generate_special_text(image_path, args, features, chars)
        ratingtag = max(rating, key=rating.get)
        wd14_caption = wd14_caption + ', ' + boorutag + f", rating:{ratingtag}"
        wd14_caption = format_wd14_caption(wd14_caption)
        aestag = get_aesthetic_tag(image)
        more_detailed_caption, firsttag = generate_response(tokenizer, model, image_processor, image, wd14_caption + ', ' + aestag, chartags)
        aestag = f"{aestag} " if 'bad' in aestag else ''
        tags_text = f"{keeptag}, {firsttag}, {aestag}___{more_detailed_caption}"
        if args.enable_wildcard:
            tags = [tag.strip() for tag in more_detailed_caption.split(',')]            
            for threshold in [round(x * 0.1, 1) for x in range(9, -1, -1)]:
                selected_tags = [tag for tag in tags if random.random() < threshold]
                additional_tags_text = f"\n{keeptag}, {firsttag}, {aestag}___{', '.join(selected_tags)}"
                tags_text += additional_tags_text

        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower()) 

    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        traceback.print_exc()

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    all_final_scores = []
    for root, dirs, files in os.walk(directory):
        image_paths = []
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                process_image(image_path, args)  
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--enable_wildcard", action="store_true", help="wildcard多行")
    parser.add_argument("--low_vram", action="store_true", help="使用1.6Bmoe模型")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    find_and_process_images(args.directory, args)

