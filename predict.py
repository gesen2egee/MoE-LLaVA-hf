import argparse
import subprocess
import os
import base64
import re
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
    'moellava'
]

for lib in libraries:
    install_and_import(lib)


from io import BytesIO
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timedelta
import torch
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from openai import OpenAI

# 嘗試導入 dghs-imgutils[gpu]，如果未安裝，則跳過相關處理
try:
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags
    from imgutils.validate import anime_dbrating
except ImportError:
    print("正在安裝 dghs-imgutils[gpu]...")
    subprocess.run(["pip", "install", "dghs-imgutils[gpu]"], check=True)
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags
    from imgutils.validate import anime_dbrating
    
# 設置 MoE-LLaVA 模型參數
#MOE_MODEL_PATH = 'LanguageBind/MoE-LLaVA-StableLM-1.6B-4e-384'
MOE_MODEL_PATH = 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e' #VRAM >= 16G
MOE_DEVICE = 'cuda'

def initialize_moe_model(model_path=MOE_MODEL_PATH, device=MOE_DEVICE):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)
    return tokenizer, model, processor

def process_moe_image(image, model, tokenizer, processor):
    image_tensor = processor['image'].preprocess(Image.open(image).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
    conv_mode = "phi"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + + '\nWhat happens in this image? What does this image appears to be? How about this image?'
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id).cuda()  # 创建注意力掩码
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 传递注意力掩码
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=200,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,  # 设置 pad_token_id
                stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip().replace('\n', ' ')
    return outputs

def process_features(features: dict) -> (dict, str):
    """
    處理features字典，移除指定模式的鍵值對並生成keep_tags字串。
    
    參數:
    features (dict): 包含特徵的字典。

    返回:
    (dict, str): 返回處理後的features字典和keep_tags字串。
    """
    patterns_to_keep = [
        r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$', 
        r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', 
        r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
        r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^realistic$', 
        r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
        r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$', 
        r'^.*photo.*$', r'^tegaki$', r'^sketch$', r'^silhouette$', r'^web_address$', r'^.*border$'
    ]
    keep_tags_set = set()

    keys = list(features.keys())
    keys_to_delete = []

    for pattern in patterns_to_keep:
        regex = re.compile(pattern)
        for key in keys:
            if regex.match(key):
                keep_tags_set.add(key.replace('_', ' '))
                keys_to_delete.append(key)
    
    for key in keys_to_delete:
        if key in features:
            del features[key]
    
    keep_tags = ', '.join(sorted(keep_tags_set)).rstrip(', ')
    
    return features, keep_tags

def generate_special_text(image_path, folder_name, args, features=None, chars=None):
    """
    根據 features, image_path 和 parent_folder 生成 special_text。
    """
    base_file_name = os.path.splitext(image_path)[0]
    boorutag_path = None
    boorutag = ""
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
    if folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        if not args.not_char:
            chartag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip()
            chartags.add(chartag_from_folder)
        else:
            concept_tag = f"in this image, you can see concept of {parent_folder.split('_')[1].replace('_', ' ').strip()}. "
            
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
    if chartag_from_folder and features and "solo" in features:
        return f"a character {chartag_from_folder} in this image", boorutag
    
    if not chartag_from_folder and features and chartags and "solo" in features:
        return f"{concept_tag} a character {' '.join(chartags)} in this image" if chartags else "", boorutag
    
    if chartags:
        if len(chartags) == 1:
            chartags.append('anonamos')    
        return f'{concept_tag}the characters in this image are {" and ".join(chartags)}', boorutag
    
    return f'{concept_tag}{chartag_from_folder}', boorutag

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

def process_image(image_path, args):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """
    folder_name = args.folder_name
    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')
    
    # 檢查文件最後修改時間，如果在一周內則略過
    if tag_file_path.exists() and not args.force:
        last_modified_time = datetime.fromtimestamp(tag_file_path.stat().st_mtime)
        if datetime.now() - last_modified_time < timedelta(days=7):
            print(f"Skipping {tag_file_path} as it was modified within the last week.")
            return None, None, 'skipped'
    
    try:
        image_resize = resize_image(image_path)

        # 使用 imgutils 獲取圖片等級
        if args.moe:
            moe_caption = process_moe_image(image_resize, moe_model, moe_tokenizer, moe_processor)
            if args.caption_style != 'pure':
                rating, features, chars = get_wd14_tags(image_resize, character_threshold=0.7, general_threshold=0.2682, model_name="ConvNext_v3", drop_overlap=True)
                features, keep_tags = process_features(drop_blacklisted_tags(features))
                #if features and "solo" in features:
                #    features = drop_basic_character_tags(features)
                wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
                rating = max(rating, key=rating.get)
            if args.caption_style == 'mixed':
                tags_text = (
                    f"the whole image consists of the following: |||{wd14_caption}|||, {moe_caption}\n"
                    f"{moe_caption}\n{moe_caption}"
                    #+ ('\n' if folder_name else '')
                    + ('\n' if "solo" in features else '')
                )
            elif args.caption_style == 'wildcards':
                tags_text = f"{moe_caption}\n{wd14_caption}"
            elif args.caption_style == 'pure':
                tags_text = f"{moe_caption}"               
            else:
                if rating in ['general', 'sensitive']:
                    tags_text = moe_caption
                else:
                    tags_text = wd14_caption
        else:
            rating, features, chars = get_wd14_tags(image_resize, character_threshold=0.7, general_threshold=0.2682, model_name=ConvNext_v3,drop_overlap=True)
            features = drop_basic_character_tags(features)
            tags_text = tags_to_text(features, use_escape=True, use_spaces=True)

        special_text, boorutag = generate_special_text(image_path, folder_name, args, features, chars)
        if rating:
            special_text += f", rating:{rating}"
        if keep_tags:
            special_text += f", {keep_tags}"
        tags_lines = tags_text.split('\n')
        tags_text = '\n'.join(f"{special_text}, {line}" for line in tags_lines)
        #tags_text = tags_text.replace("|||,",f"{boorutag}|||,")

        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower())        
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")

def find_and_process_images(directory, args):
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

    for image_path in tqdm(image_paths, desc="處理圖片"):
        try:
            process_image(image_path, args)
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")

def convert_path_format(directory):
    """
    转换路径格式为WSL路径格式
    """
    directory = directory.replace('\\', '/')
    if directory[1:3] == ':/':
        drive_letter = directory[0].lower()
        return directory.replace(f"{directory[0]}:/", f"/mnt/{drive_letter}/")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="啟用特殊資料夾名稱處理")
    parser.add_argument("--moe", action="store_true", help="使用 MoE-LLaVA 模型處理 general 和 sensitive 圖片")
    parser.add_argument("--force", action="store_true", help="強迫打標")
    parser.add_argument("--not_char", action="store_true", help="非角色")
    parser.add_argument("--caption_style", type=str, choices=["rating", "mixed", "wildcards", "pure"], default="mixed", help="指定圖片描述的風格")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()

    if args.moe:
        moe_tokenizer, moe_model, moe_processor = initialize_moe_model()

    directory = convert_path_format(args.directory)
    find_and_process_images(directory, args)
