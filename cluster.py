import re
import math
from typing import List, Dict, Optional, Tuple, Set
import glob
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import subprocess
import os
import base64
from io import BytesIO
import zipfile
from datetime import datetime

# 自動安裝所需庫
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.run(["pip", "install", package], check=True)
        __import__(package)

libraries = [
    'tqdm',
    'openai',
    'PIL',
    'pathlib',
    'datetime',
    'matplotlib',
    'natsort',
    'pandas'
]


for lib in libraries:
    install_and_import(lib)
    
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from openai import OpenAI
from natsort import natsorted
import pandas as pd


try:
    import cv2
except ImportError:    
    subprocess.run(["pip", "install", "Opencv-python"], check=True)
    import cv2


try:
    from imgutils.validate import anime_dbrating
except ImportError:
    print("正在安装 dghs-imgutils[gpu]...")
    subprocess.run(["pip", "install", "dghs-imgutils[gpu]", "--upgrade"], check=True)
    from imgutils.validate import anime_dbrating


# 常量
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# 設置 GPT-4 API 相關參數
GPT4O_API_KEY = "GPT4O_API_KEY"
MODEL = "gpt-4o"
client = OpenAI(api_key=GPT4O_API_KEY)

# 內建的外表標籤名單image_base_names
appearance_tags = {
    'long hair', 'breasts', 'short hair', 'blue eyes', 'large breasts', 'blonde hair', 'brown hair', 'black hair', 'hair ornament', 'red eyes', 'hat', 'bow', 'animal ears', 'ribbon', 'hair between eyes', 'very long hair', 'twintails', 'medium breasts', 'brown eyes', 'green eyes', 'blue hair', 'purple eyes', 'tail', 'yellow eyes', 'white hair', 'pink hair', 'grey hair', 'ahoge', 'braid', 'hair ribbon', 'purple hair', 'ponytail', 'multicolored hair', 'sidelocks', 'hair bow', 'earrings', 'red hair', 'small breasts', 'hairband', 'horns', 'wings', 'green hair', 'glasses', 'pointy ears', 'hairclip', 'medium hair', 'fang', 'dark skin', 'cat ears', 'blunt bangs', 'hair flower', 'pink eyes', 'hair bun', 'mole', 'hair over one eye', 'rabbit ears', 'orange hair', 'black eyes', 'two-tone hair', 'streaked hair', 'huge breasts', 'halo', 'red bow', 'twin braids', 'side ponytail', 'animal ear fluff', 'red ribbon', 'aqua eyes', 'dark-skinned female', 'parted bangs', 'two side up', 'v-shaped eyebrows', 'grey eyes', 'orange eyes', 'cat tail', 'symbol-shaped pupils', 'eyelashes', 'lips', 'black headwear', 'mole under eye', 'fox ears', 'maid headdress', 'shiny skin', 'fake animal ears', 'black bow', 'single braid', 'neck ribbon', 'black ribbon', 'gradient hair', 'double bun', 'floating hair', 'aqua hair', 'colored skin', 'swept bangs', 'facial hair', 'heterochromia', 'white headwear', 'blue bow', 'fox tail', 'witch hat', 'low twintails', 'one side up', 'headband', 'horse ears', 'beret', 'wavy hair', 'fangs', 'headphones', 'hair intakes', 'facial mark', 'thick eyebrows', 'horse girl', 'headgear', 'muscular male', 'heart-shaped pupils', 'bob cut', 'drill hair', 'sunglasses', 'dark-skinned male', 'light brown hair', 'wolf ears', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'demon girl', 'cat girl', 'mob cap', 'magical girl', 'eyes visible through hair', 'demon horns', 'single hair bun', 'high ponytail', 'x hair ornament', 'fox girl', 'blue ribbon', 'grabbing another\'s breast', 'antenna hair', 'hat ribbon', 'crown', 'pink bow', 'spiked hair', 'bat wings', 'ear piercing', 'slit pupils', 'bright pupils', 'monster girl', 'rabbit tail', 'tassel', 'head wings', 'short twintails', 'messy hair', 'horse tail', 'straight hair', 'feathered wings', 'hat bow', 'multiple tails', 'extra ears', 'eyewear on head', 'demon tail', 'dog ears', 'pale skin', 'red headwear', 'white ribbon', 'between breasts', 'colored inner hair', 'hair over shoulder', 'skin fang', 'mole under mouth', 'side braid', 'third eye', 'scar on face', 'baseball cap', 'beard', 'blue headwear', 'peaked cap', 'glowing eyes', 'white pupils', 'semi-rimless eyewear', 'low ponytail', 'twin drills', 'yellow bow', 'wolf tail', 'eyeshadow', 'french braid', 'no headwear', 'tokin hat', 'crossed bangs', 'black wings', 'green bow', 'single horn', 'dragon horns', 'drinking glass', 'hair scrunchie', 'santa hat', 'pink ribbon', 'half updo', 'freckles', 'demon wings', 'topless male', 'single earring', 'low-tied long hair', 'white skin', 'hair rings', 'mature male', 'unworn headwear', 'mole on breast', 'black-framed eyewear', 'short ponytail', 'purple bow', 'round eyewear', 'angel wings', 'goggles on head', 'braided ponytail', 'red-framed eyewear', 'curly hair', 'raised eyebrows', 'hat ornament', 'dragon girl', 'faceless male', 'asymmetrical hair', 'dog tail', 'yellow ribbon', 'top hat', 'sun hat', 'furry female', 'white hairband', 'asymmetrical bangs', 'fake tail', 'blood on face', 'star hair ornament', 'under-rim eyewear', 'white wings', 'mature female', 'multicolored eyes', 'colored eyelashes', 'rabbit girl', 'hoop earrings', 'bouncing breasts', 'unworn hat', 'tentacle hair', 'eyebrows hidden by hair', 'green headwear', 'wolf girl', 'light blue hair', 'mini hat', 'military hat', 'brown headwear', 'dragon tail', 'striped bow', 'tress ribbon', 'pink lips', 'short eyebrows', 'scar across eye', 'mustache', 'folded ponytail', 'dog girl', 'furry male', 'blue skin', 'heart hair ornament', 'muscular female', 'red hairband', 'hime cut', 'mouse ears', 'bandaid on face', 'nurse cap', 'purple ribbon', 'butterfly hair ornament', 'straw hat', 'green ribbon', 'visor cap', 'orange bow', 'stud earrings', 'licking lips', 'bags under eyes', 'low wings', 'long bangs', 'eyeliner', 'red lips', 'fake horns', 'back bow', 'crown braid', 'tail ornament', 'hanging breasts', 'sailor hat', 'hair behind ear', 'cabbie hat', 'flipped hair', 'single side bun', 'absurdly long hair', 'frog hair ornament', 'on head', 'fairy wings', 'star-shaped pupils', 'bird wings', 'hair over eyes', 'cow ears', 'glass', 'food-themed hair ornament', 'pink headwear', 'wrist scrunchie', 'black horns', 'headdress', 'feather hair ornament', 'tinted eyewear', 'ringed eyes', 'mask on head', 'covered eyes', 'horn ornament', 'cow horns', 'mini crown', 'very short hair', 'blue hairband', 'green skin', 'blue halo', 'tiger ears', 'symbol in eye', 'wet hair', 'purple headwear', 'flat cap', 'wine glass', 'snake hair ornament', 'cone hair bun', 'curled horns', 'ice wings', 'bald', 'mechanical halo', 'red horns', 'animal hat', 'raccoon ears', 'pink halo', 'unworn eyewear', 'lolita hairband', 'star earrings', 'crescent hair ornament', 'mouse tail', 'leg ribbon', 'garrison cap', 'white eyes', 'deep skin', 'frilled bow', 'tilted headwear', 'animal on head', 'grey skin', 'ear ornament', 'asymmetrical wings', 'two tails', 'facial tattoo', 'crescent hat ornament', 'rolling eyes', 'toned male', 'no pupils', 'glowing eye', 'fish tail', 'constricted pupils', 'split-color hair', 'leaf hair ornament', 'rabbit hair ornament', 'red skin', 'chest hair', 'leaf on head', 'goat horns', 'necktie between breasts', 'raccoon tail', 'multicolored skin', 'polka dot bow', 'ears through headwear', 'purple skin', 'heart earrings', 'double-parted bangs', 'dark blue hair', 'big hair', 'frilled hairband', 'hair over breasts', 'blank eyes', 'lion ears', 'sparkling eyes', 'tiger tail', 'cow girl', 'huge ahoge', 'tassel earrings', 'star hat ornament', 'braided bun', 'assertive female', 'grey headwear', 'mini top hat', 'arm ribbon', 'braided bangs', 'bear ears', 'shark tail', 'red halo', 'red eyeshadow', 'sheep horns', 'insect wings', 'rimless eyewear', 'bow hairband', 'skin-covered horns', 'yellow halo', 'anchor hair ornament', 'navel hair', 'yellow hairband', 'no eyes', 'ear bow', 'gigantic breasts', 'extra eyes', 'long braid', 'jphones', 'large bow', 'tail ribbon', 'bird ears', 'pink skin', 'cat boy', 'shark girl', 'mouse girl', 'arthropod girl', 'fur hat', 'fur-trimmed headwear', 'raised eyebrow', 'black skin', 'frilled hat', 'striped ribbon', 'waist bow', 'super crown', 'low twin braids', 'crazy eyes', 'cat hair ornament', 'blue wings', 'naked ribbon', 'butterfly wings', 'multiple hair bows', 'demon boy', 'sagging breasts', 'dress bow', 'red scrunchie', 'dragon wings', 'forked eyebrows', 'armpit hair', 'footwear bow', 'purple hairband', 'multiple wings', 'wrist ribbon', 'v over eye', 'red pupils', 'pirate hat', 'towel on head', 'orange headwear', 'bow-shaped hair', 'against glass', 'leg hair', 'mini wings', 'multiple horns', 'carrot hair ornament', 'long eyelashes', 'backwards hat', 'black tail', 'red headband', 'tiger girl', 'mechanical wings', 'white horns', 'musical note hair ornament', 'unaligned breasts', 'orange ribbon', 'heart-shaped eyewear', 'small horns', 'uneven eyes', 'lion tail', 'dangle earrings', 'print bow', 'dog boy', 'raccoon girl', 'blue scrunchie', 'lion girl', 'opaque glasses', 'robot ears', 'christmas ornaments', 'biting own lip', 'framed breasts', 'wizard hat', 'cat ear headphones', 'quad tails', 'bandage over one eye', 'sheep ears', 'arms under breasts', 'diagonal bangs', 'wing hair ornament', 'perky breasts', 'bone hair ornament', 'striped tail', 'cuts', 'medical eyepatch', 'braided hair rings', 'multicolored wings', 'rectangular eyewear', 'purple wings', 'squirrel ears', 'ear ribbon', 'black headband', 'multiple earrings', 'single hair intake', 'sheep girl', 'updo', 'bat hair ornament', 'goggles on headwear', 'horned headwear', 'white scrunchie', 'red eyeliner', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'squirrel tail', 'horn bow', 'green hairband', 'horizontal pupils', 'stained glass', 'wolf boy', 'horseshoe ornament', 'chef hat', 'black lips', 'fox boy', 'multi-tied hair', 'slime girl', 'animal ear piercing', 'shark hair ornament', 'bird girl', 'gold earrings', 'tassel hair ornament', 'feather hair', 'puckered lips', 'orange hairband', 'ankle ribbon', 'flower earrings', 'grey horns', 'crescent earrings', 'yellow pupils', 'drill sidelocks', 'pink scrunchie', 'strap between breasts', 'winged hat', 'ghost tail', 'porkpie hat', 'parted hair', 'squirrel girl', 'police hat', 'over-rim eyewear', 'diagonal-striped bow', 'shower head', 'monkey tail', 'energy wings', 'wide ponytail', 'snowflake hair ornament', 'yellow scrunchie', 'brown ribbon', 'jackal ears', 'bandaged head', 'high side ponytail', 'blue lips', 'clover hair ornament', 'diamond-shaped pupils', 'long pointy ears', 'frilled ribbon', 'broken glass', 'flame-tipped tail', 'turning head', 'tiger boy', 'hair horns', 'skin fangs', 'deer ears', 'looking over eyewear', 'pink-framed eyewear', 'feather earrings', 'broken horn', 'laurel crown', 'large hat', 'flaming eye', 'pom pom hair ornament', 'grey bow', 'disembodied head', 'narrowed eyes', 'no eyewear', 'yellow skin', 'orange scrunchie', 'aqua ribbon', 'large tail', 'averting eyes', 'dreadlocks', 'character hair ornament', 'mechanical horns', 'grey-framed eyewear', 'star halo', 'cocktail glass', 'striped horns', 'multiple moles', 'curtained hair', 'cat hat', 'green lips', 'shako cap', 'buzz cut', 'dragon boy', 'alternate headwear', 'asymmetrical horns', 'short bangs', 'orange-tinted eyewear', 'cracked skin', 'yellow-framed eyewear', 'bandage on face', 'snake tail', 'thigh ribbon', 'afro', 'white-framed eyewear', 'd-pad hair ornament', 'tri tails', 'spread wings', 'school hat', 'tall female', 'bisexual female', 'cone horns', 'pink pupils', 'hair through headwear', 'mechanical tail', 'prehensile hair', 'patchwork skin', 'blue eyeshadow', 'drop earrings', 'veiny breasts', 'two-tone ribbon', 'bear hair ornament', 'bowl hat', 'gold hairband', 'spider girl', 'red-tinted eyewear', 'eyebrow cut', 'animal ear headwear', 'goat ears', 'single hair ring', 'fish hair ornament', 'dixie cup hat', 'leopard ears', 'skull earrings', 'party hat', 'blue horns', 'brushing hair', 'plaid headwear', 'white tail', 'brown hairband', 'blood from eyes', 'fiery hair', 'green halo', 'dyed bangs', 'two-tone eyes', 'wrinkled skin', 'bat ears', 'black halo', 'upturned eyes', 'bowl cut', 'bear girl', 'blue headband', 'yellow wings', 'fish girl', 'fake wings', 'x-shaped pupils', 'fake facial hair', 'flower ornament', 'pillbox hat', 'circle cut', 'yellow horns', 'body hair', 'hair ears', 'bow earrings', 'no wings', 'doughnut hair bun', 'green-framed eyewear', 'magnifying glass', 'eyewear on headwear', 'brown horns', 'plant girl', 'pink eyeshadow', 'multiple braids', 'magatama earrings', 'brown-framed eyewear', 'blue-tinted eyewear', 'cow boy', 'spiked tail', 'purple eyeshadow', 'body freckles', 'multicolored bow', 'heart tail', 'large wings', 'triangle earrings', 'rabbit boy', 'horns through headwear', 'purple-tinted eyewear', 'unusually open eyes', 'sunflower hair ornament', 'bruise on face', 'lizard tail', 'multicolored horns', 'arm between breasts', 'two-tone headwear', 'panda ears', 'fake mustache', 'expressive hair', 'purple tail', 'drawing bow', 'object through head', 'pink wings', 'blue pupils', 'transparent wings', 'purple horns', 'phoenix crown', 'artificial eye', 'grey ribbon', 'striped headwear', 'goat girl', 'tulip hat', 'crystal hair', 'aqua headwear', 'arched bangs', 'broken halo', 'mechanical ears', 'brown wings', 'leopard tail', 'grey halo', 'no eyebrows', 'notched ear', 'monkey ears', 'pink-tinted eyewear', 'fiery horns', 'uneven horns', 'jaguar ears', 'purple halo', 'sphere earrings', 'bat girl', 'candy hair ornament', 'brushing another\'s hair', 'tapir tail', 'dark halo', 'ruffling hair', 'diving mask on head', 'triangle hair ornament', 'mechanical eye', 'huge bow', 'robot girl', 'sleeve bow', 'rabbit-shaped pupils', 'dice hair ornament', 'button eyes', 'chocolate on breasts', 'prehensile tail', 'multicolored headwear', 'green wings', 'looking at breasts', 'solid eyes', 'thick lips', 'compass rose halo', 'brown tail', 'strawberry hair ornament', 'food-themed earrings', 'split ponytail', 'two-tone bow', 'neck tassel', 'lion boy', 'two-tone hairband', 'gradient skin', 'polka dot headwear', 'purple scrunchie', 'glowing wings', 'crystal earrings', 'liquid hair', 'orange skin', 'cetacean tail', 'glowing hair', 'smokestack hair ornament', 'panties on head', 'crocodilian tail', 'long tail', 'legs over head', 'pearl earrings', 'glowing horns', 'red tail', 'print headwear', 'egg hair ornament', 'side drill', 'blue tail', 'huge eyebrows', 'hair wings', 'snake hair', 'thick eyelashes', 'swim cap', 'grey tail', 'choppy bangs', 'aviator sunglasses', 'pill earrings', 'no tail', 'pink tail', 'owl ears', 'pointy breasts', 'hat over one eye', 'full beard', 'bandaid hair ornament', 'footwear ribbon', 'grey hairband', 'coin hair ornament', 'bucket hat', 'alpaca ears', 'yellow tail', 'low-tied sidelocks', 'weasel ears', 'wrist bow', 'grey wings', 'pursed lips', 'no eyepatch', 'deer girl', 'white headdress', 'green tail', 'wing ornament', 'mismatched eyebrows', 'sleeve ribbon', 'purple-framed eyewear', 'rainbow hair', 'hedgehog ears', 'sideways hat', 'flower on head', 'coke-bottle glasses', 'fish boy', 'orange tail', 'hard hat', 'hair on horn', 'ribbon-trimmed headwear', 'multiple heads', 'flower over eye', 'yellow-tinted eyewear', 'otter ears', 'dashed eyes', 'low-braided long hair', 'arm above head', 'lace-trimmed hairband', 'four-leaf clover hair ornament', 'potara earrings', 'detached hair', 'cephalopod eyes', 'long beard', 'camouflage headwear', 'japari bun', 'star ornament', 'striped hairband', 'hat with ears', 'bunching hair', 'ears visible through hair', 'green scrunchie', 'thick mustache', 'diamond hairband', 'polka dot scrunchie', 'cherry hair ornament', 'bear tail', 'jaguar tail', 'v-shaped eyes', 'rabbit hat', 'thick beard', 'hugging tail', 'no mole', 'green-tinted eyewear', 'ornament', 'diamond hair ornament', 'wavy eyes', 'shell hair ornament', 'heart-shaped eyes', 'chain headband', 'planet hair ornament', 'pearl hair ornament', 'multicolored hairband', 'drop-shaped pupils', 'polka dot ribbon', 'ribbon braid', 'alternate wings', 'hollow eyes', 'unworn eyepatch', 'food on breasts', 'spaceship hair ornament', 'bowler hat', 'green eyeshadow', 'pumpkin hair ornament', 'spiked hairband', 'flower in eye', 'magical boy', 'behind-the-head headphones', 'plaid ribbon', 'skull ornament', 'bear boy', 'holly hair ornament', 'uneven twintails', 'folded hair', 'pig ears', 'metal skin', 'pumpkin hat', 'cut bangs', 'mole under each eye', 'clock eyes', 'reptile girl', 'hair between breasts', 'alternate hair ornament', 'licking ear', 'braiding hair', 'hexagon hair ornament', 'tri braids', 'animal ear hairband', 'clothed male nude male', 'penis over eyes', 'solid circle pupils', 'penis to breast', 'frog girl', 'curly eyebrows', 'star-shaped eyewear', 'fiery wings', 'orange headband', 'scratching head', 'bloodshot eyes', 'green horns', 'green headband', 'single head wing', 'animal head', 'bulging eyes', 'deer tail', 'weasel girl', 'brown lips', 'lifebuoy ornament', 'frilled headwear', 'cable tail', 'safety glasses', 'leopard girl', 'wing ears', 'spade hair ornament', 'white halo', 'weasel tail', 'propeller hair ornament', 'wide oval eyes', 'otter tail', 'pom pom earrings', 'checkered bow', 'fruit hat ornament', 'starfish hair ornament', 'aqua hairband', 'crystal wings', 'object head', 'multicolored tail', 'gradient wings', 'giant male', 'purple pupils', 'torn wings', 'head on head', 'moose ears', 'pointy hat', 'hair over one breast', 'arm over head', 'grabbing another\'s ear', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'hedgehog tail', 'lop rabbit ears', 'sparse chest hair', 'pink horns', 'pokemon ears', 'ankle bow', 'bird boy', 'bandaid on head', 'implied extra ears', 'hat tassel', 'fruit on head', 'starry hair', 'sparkle hair ornament', 'long ribbon', 'rice hat', 'washing hair', 'anchor earrings', 'asymmetrical sidelocks', 'mini witch hat', 'unworn hair ornament', 'heart hair', 'arthropod boy', 'detached ahoge', 'large ears', 'aviator cap', 'monkey boy', 'female service cap', 'moth girl', 'glove bow', 'bangs', 'shiny hair', 'light purple hair', 'oni horns', 'pillow hat', 'polos crown', 'light green hair', 'monocle hair ornament', 'dark green hair', 'pouty lips', 'bunny-shaped pupils', 'bunny hatester cap', 'detached wings', 'solid oval eyes', 'cube hair ornament', 'heart ahoge', 'cross-shaped pupils', 'cross hair ornament', 'pointy hair', 'very dark skin', 'aqua bow', 'front ponytail', 'pink hairband', 'skull hair ornament', 'side braids', 'tail bow', 'cross earrings', 'horn ribbon', 'cow tail', 'floppy ears', 'two-tone skin', 'plaid bow', 'purple lips', 'single sidelock', 'solid circle eyes', 'yellow headwear', 'faceless female', 'single wing', 'brown bow', 'medium bangs', 'red wings', 'monster boy', 'mismatched pupils', 'cowboy hat', 'flower-shaped pupils', 'bird tail', 'gradient eyes', 'bursting breasts', 'animal ear head'
}

# 內建的衣服標籤名單
clothing_tags = {
    'shirt' , 'skirt' , 'long sleeves' , 'hair ornament' , 'gloves' , 'dress' , 'thighhighs' , 'hat' , 'jewelry' , 'underwear' , 'jacket' , 'school uniform' , 'white shirt' , 'panties' , 'swimsuit' , 'hair ribbon' , 'short sleeves' , 'hair bow' , 'pantyhose' , 'earrings' , 'bikini' , 'pleated skirt' , 'frills' , 'hairband' , 'boots' , 'open clothes' , 'necktie' , 'detached sleeves' , 'shorts' , 'japanese clothes' , 'shoes' , 'sleeveless' , 'black gloves' , 'collared shirt' , 'choker' , 'socks' , 'glasses' , 'pants' , 'serafuku' , 'puffy sleeves' , 'hairclip' , 'belt' , 'black thighhighs' , 'elbow gloves' , 'white gloves' , 'bowtie' , 'hood' , 'black skirt' , 'tongue out' , 'wide sleeves' , 'miniskirt' , 'fingerless gloves' , 'black footwear' , 'armpits' , 'kimono' , 'white dress' , 'holding weapon' , 'off shoulder' , 'necklace' , 'striped clothes' , 'nail polish' , 'bag' , 'black dress' , 'scarf' , 'cape' , 'white thighhighs' , 'bra' , 'armor' , 'vest' , 'open jacket' , 'apron' , 'red bow' , 'white panties' , 'leotard' , 'coat' , 'black jacket' , 'high heels' , 'collar' , 'sweater' , 'bracelet' , 'red ribbon' , 'crop top' , 'black shirt' , 'puffy short sleeves' , 'blue skirt' , 'fingernails' , 'black pantyhose' , 'neckerchief' , 'sleeves past wrists' , 'fur trim' , 'see-through' , 'wrist cuffs' , 'maid' , 'zettai ryouiki' , 'clothing cutout' , 'black headwear' , 'plaid' , 'torn clothes' , 'mole under eye' , 'one-piece swimsuit' , 'sash' , 'maid headdress' , 'sleeveless shirt' , 'short shorts' , 'sleeveless dress' , 'ascot' , 'black panties' , 'cosplay' , 'kneehighs' , 'thigh strap' , 'black bow' , 'hoodie' , 'neck ribbon' , 'black ribbon' , 'black choker' , 'dress shirt' , 'buttons' , 'open shirt' , 'sideboob' , 'mask' , 'capelet' , 'bodysuit' , 'blue dress' , 'black pants' , 'black bikini' , 'white headwear' , 'red skirt' , 'blue bow' , 'turtleneck' , 'underboob' , 'witch hat' , 'highleg' , 'military uniform' , 'headband' , 'black shorts' , 'bottomless' , 'beret' , 'side-tie bikini bottom' , 'brown footwear' , 'halterneck' , 'playboy bunny' , 'piercing' , 'white jacket' , 'holding sword' , 'white socks' , 'chinese clothes' , 'plaid skirt' , 'thigh boots' , 'white footwear' , 'headgear' , 'sandals' , 'muscular male' , 'floral print' , 'garter straps' , 'short dress' , 'sunglasses' , 'obi' , 'red dress' , 'hood down' , 'frilled dress' , 'cleavage cutout' , 'white skirt' , 'blue shirt' , 'ring' , 'holding food' , 'blue jacket' , 'black socks' , 'black hairband' , 'white flower' , 'white bow' , 'formal' , 'topless' , 'mob cap' , 'cardigan' , 'pantyshot' , 'frilled skirt' , 'tank top' , 'blazer' , 'suspenders' , 'helmet' , 'suit' , 'x hair ornament' , 'underwear only' , 'blue ribbon' , 'frilled sleeves' , 'school swimsuit' , 'hat ribbon' , 'denim' , 'crown' , 'knee boots' , 'red necktie' , 'tiara' , 'breasts out' , 'red flower' , 'juliet sleeves' , 'polka dot' , 'lingerie' , 'animal print' , 'red shirt' , 'undressing' , 'striped thighhighs' , 'blue sailor collar' , 'sneakers' , 'black leotard' , 'white border' , 't-shirt' , 'tassel' , 'holding gun' , 'red footwear' , 'white apron' , 'tan' , 'red bowtie' , 'hair bobbles' , 'lipstick' , 'green skirt' , 'goggles' , 'shoulder armor' , 'holding cup' , 'brooch' , 'black bra' , 'fishnets' , 'loafers' , 'towel' , 'single thighhigh' , 'pink dress' , 'strapless leotard' , 'hat bow' , 'grey shirt' , 'black necktie' , 'eyewear on head' , 'bike shorts' , 'hooded jacket' , 'armband' , 'casual' , 'revealing clothes' , 'red headwear' , 'white ribbon' , 'china dress' , 'ribbon trim' , 'pink panties' , 'multicolored clothes' , 'wristband' , 'hakama' , 'blouse' , 'puffy long sleeves' , 'veil' , 'red jacket' , 'lace trim' , 'scar on face' , 'waist apron' , 'skirt set' , 'large pectorals' , 'pink flower' , 'pelvic curtain' , 'strapless dress' , 'baseball cap' , 'string bikini' , 'striped panties' , 'blue headwear' , 'bridal gauntlets' , 'cloak' , 'peaked cap' , 'highleg leotard' , 'red neckerchief' , 'purple dress' , 'side-tie panties' , 'semi-rimless eyewear' , 'white pantyhose' , 'grey skirt' , 'front-tie top' , 'bow panties' , 'buckle' , 'pom pom \(clothes\)' , 'clothing aside' , 'micro bikini' , 'yellow bow' , 'maid apron' , 'sleeves past fingers' , 'hood up' , 'corset' , 'skin tight' , 'hakama skirt' , 'black belt' , 'lace' , 'holding phone' , 'tokin hat' , 'white sleeves' , 'cropped jacket' , 'bikini top only' , 'brown gloves' , 'red gloves' , 'mary janes' , 'full moon' , 'blue bikini' , 'holding book' , 'side slit' , 'black coat' , 'camisole' , 'armlet' , 'green bow' , 'hair scrunchie' , 'sleeves rolled up' , 'gold trim' , 'blue necktie' , 'santa hat' , 'black sailor collar' , 'pink skirt' , 'single glove' , 'pink ribbon' , 'white sailor collar' , 'zipper' , 'blue flower' , 'open coat' , 'blue shorts' , 'pencil skirt' , 'pink shirt' , 'ribbed sweater' , 'topless male' , 'high heel boots' , 'frilled apron' , 'asymmetrical legwear' , 'sweater vest' , 'cross-laced footwear' , 'upskirt' , 'black vest' , 'frilled bikini' , 'pocket' , 'green dress' , 'unworn headwear' , 'frilled shirt collar' , 'long fingernails' , 'black-framed eyewear' , 'brown pantyhose' , 'thong' , 'red bikini' , 'purple bow' , 'long skirt' , 'high-waist skirt' , 'round eyewear' , 'crying with eyes open' , 'blue footwear' , 'white bra' , 'gym uniform' , 'purple skirt' , 'yellow shirt' , 'goggles on head' , 'black bowtie' , 'braided ponytail' , 'red-framed eyewear' , 'epaulettes' , 'ribbon-trimmed sleeves' , 'santa costume' , 'brown jacket' , 'tanlines' , 'denim shorts' , 'brown skirt' , 'hat ornament' , 'panties under pantyhose' , 'buruma' , 'white pants' , 'nontraditional miko' , 'pouch' , 'thighband pantyhose' , 'black serafuku' , 'red scarf' , 'green jacket' , 'sailor dress' , 'robe' , 'garter belt' , 'white shorts' , 'competition swimsuit' , 'yellow ribbon' , 'lolita fashion' , 'top hat' , 'sun hat' , 'white hairband' , 'watch' , 'blood on face' , 'blue one-piece swimsuit' , 'turtleneck sweater' , 'star hair ornament' , 'white kimono' , 'sports bra' , 'under-rim eyewear' , 'grey jacket' , 'blurry foreground' , 'shiny clothes' , 'white coat' , 'striped shirt' , 'impossible clothes' , 'jeans' , 'yellow flower' , 'circlet' , 'belt buckle' , 'holding umbrella' , 'green shirt' , 'partially fingerless gloves' , 'sarashi' , 'striped bikini' , 'shawl' , 'bandaged arm' , 'hairpin' , 'hoop earrings' , 'sheathed' , 'purple flower' , 'unworn hat' , 'blue bowtie' , 'green headwear' , 'yukata' , 'mini hat' , 'white leotard' , 'purple shirt' , 'military hat' , 'pajamas' , 'brown headwear' , 'black sleeves' , 'blue pants' , 'bespectacled' , 'holding staff' , 'shirt tucked in' , 'striped bow' , 'tress ribbon' , 'mouth mask' , 'blue panties' , 'animal hood' , 'collared dress' , 'scar across eye' , 'backless outfit' , 'tabard' , 'adjusting clothes' , 'long dress' , 'torn pantyhose' , 'fishnet pantyhose' , 'toenail polish' , 'off-shoulder dress' , 'front-tie bikini top' , 'pink footwear' , 'heart hair ornament' , 'sharp fingernails' , 'lab coat' , 'panties aside' , 'black one-piece swimsuit' , 'red hairband' , 'skirt hold' , 'wedding dress' , 'blue kimono' , 'nurse cap' , 'purple ribbon' , 'holding clothes' , 'bloomers' , 'butterfly hair ornament' , 'red cape' , 'hat flower' , 'blue gloves' , 'miko' , 'pasties' , 'straw hat' , 'green ribbon' , 'bandana' , 'black bodysuit' , 'blue leotard' , 'visor cap' , 'winter uniform' , 'orange bow' , 'drawstring' , 'yellow ascot' , 'red vest' , 'stud earrings' , 'blindfold' , 'fur-trimmed jacket' , 'brown belt' , 'off-shoulder shirt' , 'licking lips' , 'pink jacket' , 'center frills' , 'shrug \(clothing\)' , 'panties around one leg' , 'black cape' , 'pink bra' , 'criss-cross halter' , 'anklet' , 'center opening' , 'white sweater' , 'headpiece' , 'suspender skirt' , 'highleg panties' , 'beanie' , 'grabbing from behind' , 'spaghetti strap' , 'bandeau' , 'white scarf' , 'pink bikini' , 'blanket' , 'underbust' , 'holding knife' , 'back bow' , 'pov hands' , 'white one-piece swimsuit' , 'adjusting eyewear' , 'white outline' , 'partially visible vulva' , 'double-breasted' , 'brown thighhighs' , 'gakuran' , 'geta' , 'sailor hat' , 'nun' , 'food on face' , 'black collar' , 'tabi' , 'headwear' , 'cabbie hat' , 'single side bun' , 'microskirt' , 'pinafore dress' , 'arm warmers' , 'petticoat' , 'frog hair ornament' , 'sailor shirt' , 'bodystocking' , 'highleg swimsuit' , 'frilled shirt' , 'body fur' , 'reaching towards viewer' , 'yellow jacket' , 'unbuttoned' , 'bird wings' , 'winter clothes' , 'bikini under clothes' , 'wristwatch' , 'summer uniform' , 'gothic lolita' , 'cross-laced clothes' , 'cow print' , 'brown shirt' , 'lace-up boots' , 'cheerleader' , 'checkered clothes' , 'poke ball \(basic\)' , 'holding bag' , 'grey pants' , 'brown dress' , 'swimsuit under clothes' , 'taut clothes' , 'overalls' , 'holding bottle' , 'pom pom \(cheerleading\)' , 'pink headwear' , 'wrist scrunchie' , 'black sweater' , 'mittens' , 'blue thighhighs' , 'holding poke ball' , 'military vehicle' , 'hair stick' , 'ankle boots' , 'layered sleeves' , 'toeless legwear' , 'headdress' , 'feather hair ornament' , 'tinted eyewear' , 'blood on clothes' , 'wedding ring' , 'bangle' , 'purple bikini' , 'unzipped' , 'sundress' , 'pleated dress' , 'purple jacket' , 'crotch seam' , 'armored dress' , 'armored boots' , 'mask on head' , 'midriff peek' , 'red kimono' , 'black kimono' , 'purple gloves' , 'grey dress' , 'bridal garter' , 'tube top' , 'holding tray' , 'holding fan' , 'halloween costume' , 'open fly' , 'mini crown' , 'white capelet' , 'animal costume' , 'shoulder cutout' , 'naked shirt' , 'half gloves' , 'blue hairband' , 'brown pants' , 'red choker' , 'male underwear' , 'black hoodie' , 'uneven legwear' , 'falling petals' , 'blue vest' , 'haori' , 'waitress' , 'thighlet' , 'holding polearm' , 'red thighhighs' , 'purple thighhighs' , 'open cardigan' , 'yellow bikini' , 'two-tone dress' , 'bow bra' , 'bridal veil' , 'purple headwear' , 'flat cap' , 'pilot suit' , 'paw gloves' , 'tight clothes' , 'holding microphone' , 'red panties' , 'blue neckerchief' , 'ear covers' , 'triangular headpiece' , 'snake hair ornament' , 'star print' , 'paw print' , 'pink kimono' , 'purple panties' , 'fishnet thighhighs' , 'heart of string' , 'red hakama' , 'yellow dress' , 'disposable cup' , 'black border' , 'sarong' , 'red shorts' , 'slippers' , 'mismatched legwear' , 'highleg bikini' , 'yellow neckerchief' , 'cropped shirt' , 'oil-paper umbrella' , 'hakama short skirt' , 'sleeve cuffs' , 'red ascot' , 'hip vent' , 'animal hat' , 'arm strap' , 'navel cutout' , 'scar on cheek' , 'white cape' , 'white choker' , 'partially unbuttoned' , 'unworn eyewear' , 'condom wrapper' , 'lolita hairband' , 'backless dress' , 'casual one-piece swimsuit' , 'torn thighhighs' , 'tied shirt' , 'layered dress' , 'string panties' , 'bandaged leg' , 'sleeveless turtleneck' , 'bikini skirt' , 'red leotard' , 'layered skirt' , 'leggings' , 'green bikini' , 'star earrings' , 'scabbard' , 'brown coat' , 'pantyhose under shorts' , 'tasuki' , 'invisible chair' , 'lace-trimmed panties' , 'lace-trimmed legwear' , 'leg ribbon' , 'tate eboshi' , 'plugsuit' , 'hat feather' , 'garrison cap' , 'short kimono' , 'parasol' , 'fur-trimmed coat' , 'leather' , 'asymmetrical clothes' , 'lace-trimmed bra' , 'shimenawa' , 'frilled bow' , 'tilted headwear' , 'animal on head' , 'yellow skirt' , 'sample watermark' , 'holding hair' , 'purple footwear' , 'blue bra' , 'raglan sleeves' , 'harness' , 'clothes around waist' , 'tiger print' , 'blue scarf' , 'grey footwear' , 'asymmetrical gloves' , 'ear ornament' , 'o-ring top' , 'naked towel' , 'standing sex' , 'o-ring bikini' , 'two tails' , 'white collar' , 'purple kimono' , 'white tank top' , 'cross necklace' , 'french kiss' , 'crescent hat ornament' , 'crop top overhang' , 'short over long sleeves' , 'black scarf' , 'open kimono' , 'oversized clothes' , 'black neckerchief' , 'holding bouquet' , 'fur-trimmed sleeves' , 'platform footwear' , 'one-piece tan' , 'white hoodie' , 'striped dress' , 'orange shirt' , 'blue cape' , 'holding gift' , 'pectoral cleavage' , 'white belt' , 'leaf hair ornament' , 'holding stuffed toy' , 'red pants' , 'rabbit hair ornament' , 'pink bowtie' , 'brown sweater' , 'yellow necktie' , 'holding fruit' , 'fox mask' , 'spiked bracelet' , 'unworn shoes' , 'red coat' , 'shoe soles' , 'water bottle' , 'black tank top' , 'grey shorts' , 'head scarf' , 'leaf on head' , 'micro shorts' , 'raccoon tail' , 'slingshot swimsuit' , 'multicolored skin' , 'camouflage' , 'east asian architecture' , 'polka dot bow' , 'striped necktie' , 'naked apron' , 'red capelet' , 'green vest' , 'white bowtie' , 'fur-trimmed gloves' , 'loose socks' , 'badge' , 'multicolored jacket' , 'shorts under skirt' , 'heart earrings' , 'black capelet' , 'bead necklace' , 'babydoll' , 'green necktie' , 'brown shorts' , 'undershirt' , 'frilled hairband' , 'grey sweater' , 'orange skirt' , 'pov crotch' , 'see-through sleeves' , 'hooded cloak' , 'bonnet' , 'green shorts' , 'blue choker' , 'brown cardigan' , 'gym shirt' , 'blue coat' , 'bead bracelet' , 'fundoshi' , 'white ascot' , 'cow girl' , 'hachimaki' , 'white bodysuit' , 'plaid vest' , 'pink gloves' , 'argyle clothes' , 'tassel earrings' , 'drink can' , 'bikini armor' , 'black sports bra' , 'yellow bowtie' , 'pink thighhighs' , 'star hat ornament' , 'latex' , 'holding chopsticks' , 'purple bowtie' , 'torn shirt' , 'grey headwear' , 'sweater dress' , 'bobby socks' , 'grey gloves' , 'blue sweater' , 'wardrobe malfunction' , 'green footwear' , 'striped pantyhose' , 'mini top hat' , 'arm garter' , 'aqua necktie' , 'frilled panties' , 'arm ribbon' , 'fur-trimmed capelet' , 'untied bikini' , 'holding spoon' , 'holding bow \(weapon\)' , 'red collar' , 'ear blush' , 'object on head' , 'multicolored dress' , 'single shoe' , 'bikini bottom only' , 'shark tail' , 'yellow footwear' , 'red eyeshadow' , 'green kimono' , 'vertical-striped thighhighs' , 'red sweater' , 'heart brooch' , 'rimless eyewear' , 'bow hairband' , 'male swimwear' , 'frilled bra' , 'holding fork' , 'green panties' , 'grey thighhighs' , 'zouri' , 'collared jacket' , 'race queen' , 'fur-trimmed dress' , 'traditional bowtie' , 'anchor hair ornament' , 'yellow hairband' , 'ear bow' , 'unworn panties' , 'fur-trimmed cape' , 'long coat' , 'unworn skirt' , 'blue hoodie' , 'between fingers' , 'sleeveless jacket' , 'green pants' , 'white neckerchief' , 'superhero' , 'single sock' , 'brown vest' , 'print panties' , 'waist cape' , 'green gloves' , 'holding can' , 'hooded coat' , 'jester cap' , 'open vest' , 'plaid shirt' , 'holding cigarette' , 'monocle' , 'black blindfold' , 'pink sweater' , 'track suit' , 'pink choker' , 'pillow hug' , 'cube hair ornament' , 'frilled collar' , 'floating object' , 'black suit' , 'nightgown' , 'the pose' , 'frilled choker' , 'striped bowtie' , 'beach umbrella' , 'white necktie' , 'jumpsuit' , 'uwabaki' , 'bride' , 'blood on hands' , 'downblouse' , 'holding wand' , 'kariginu' , 'cross hair ornament' , 'competition school swimsuit' , 'over-kneehighs' , 'aqua bow' , 'leotard under clothes' , 'lowleg panties' , 'pink hairband' , 'taut shirt' , 'skull hair ornament' , 'red bodysuit' , 'heart print' , 'heart cutout' , 'plaid scarf' , 'side braids' , 'holding hat' , 'open hoodie' , 'red bra' , 'purple bra' , 'adjusting headwear' , 'short necktie' , 'blood splatter' , 'cross earrings' , 'holding candy' , 'coat on shoulders' , 'halter dress' , 'horn ribbon' , 'multiple rings' , 'two-tone skin' , 'plaid bow' , 'dougi' , 'leg warmers' , 'loincloth' , 'frilled thighhighs' , 'pinstripe pattern' , 'purple necktie' , 'meat' , 'forehead protector' , 'holding paper' , 'two-tone shirt' , 'holding ball' , 'yellow headwear' , 'sleeves past elbows' , 'heart pasties' , 'white camisole' , 'thighhighs under boots' , 'belt collar' , 'brown bow' , 'bubble skirt' , 'striped socks' , 'blue butterfly' , 'waistcoat' , 'cowboy hat' , 'print dress' , 'asymmetrical sleeves' , 'see-through shirt' , 'unworn clothes' , 'polka dot panties' , 'lapels' , 'blue bodysuit' , 'torn pants' , 'star in eye' , ' eyewear' , 'bird tail' , 'holding pen' , 'grey pantyhose' , 'gradient eyes' , 'cow print bikini' , 'thong bikini' , 'bra visible through clothes' , 'white bloomers' , 'panty peek' , 'chest sarashi' , 'wet panties' , 'yellow gloves' , 'large bow' , 'old school swimsuit' , 'green bowtie' , 'blue sleeves' , 'bird ears' , 'stirrup legwear' , 'maebari' , 'orange jacket' , 'purple leotard' , 'streaming tears' , 'partially unzipped' , 'aiguillette' , 'dolphin shorts' , 'vertical-striped shirt' , 'fur hat' , 'obijime' , 'fur-trimmed headwear' , 'bra strap' , 'orange dress' , 'blue buruma' , 'pink necktie' , 'toeless footwear' , 'crotchless' , 'suit jacket' , 'striped scarf' , 'mismatched gloves' , 'grey vest' , 'blue socks' , 'medium skirt' , 'argyle legwear' , 'frilled hat' , 'tengu-geta' , 'striped skirt' , 'showgirl skirt' , 'striped ribbon' , 'waist bow' , 'red buruma' , 'falling leaves' , 'closed umbrella' , 'button gap' , 'fedora' , 'cable knit' , 'holding camera' , 'grabbing own ass' , 'backboob' , 'single leg pantyhose' , 'grey cardigan' , 'winter coat' , 'yellow sweater' , 'serval print' , 'orange flower' , 'see-through cleavage' , 'wine bottle' , 'back cutout' , 'two-tone jacket' , 'single elbow glove' , 'yellow shorts' , 'cake slice' , 'naked ribbon' , 'orange bikini' , 'black flower' , 'torn dress' , 'black ascot' , 'print skirt' , 'multiple hair bows' , 'holding broom' , 'condom in mouth' , 'unworn jacket' , 'track pants' , 'bikini bottom aside' , 'holding shield' , 'blood on weapon' , 'dress bow' , 'pink neckerchief' , 'red scrunchie' , 'grey hoodie' , 'blood from mouth' , 'business suit' , 'baggy pants' , 'holding bowl' , 'metal collar' , 'unworn mask' , 'bandaged hand' , 'white vest' , 'hose' , 'frilled hair tubes' , 'unbuttoned shirt' , 'scar on chest' , 'holding lollipop' , 'blue capelet' , 'footwear bow' , 'pocket watch' , 'purple hairband' , 'wrist ribbon' , 'single sleeve' , 'surgical mask' , 'unconventional maid' , 'open collar' , 'bustier' , 'brown bag' , 'pirate hat' , 'red rope' , 'sleeveless kimono' , 'leather jacket' , 'trench coat' , 'orange bowtie' , 'pink cardigan' , 'planted sword' , 'orange headwear' , 'loose necktie' , 'hypnosis' , 'bodypaint' , 'brown scarf' , 'lowleg bikini' , 'green sailor collar' , 'holding towel' , 'pink scarf' , 'red sailor collar' , 'oppai loli' , 'fur-trimmed boots' , 'heart in eye' , 'multicolored skirt' , 'gym shorts' , 'carrot hair ornament' , 'covered collarbone' , 'silk' , 'sailor' , 'heart necklace' , 'frilled gloves' , 'yellow panties' , 'blue ascot' , 'underboob cutout' , 'leaf print' , 'arm belt' , 'backwards hat' , 'polka dot bikini' , 'crescent pin' , 'red headband' , 'red sleeves' , 'earclip' , 'checkered skirt' , 'holding box' , 'white robe' , 'reverse outfit' , 'grey coat' , 'purple pantyhose' , 'pince-nez' , 'musical note hair ornament' , 'holding mask' , 'frilled pillow' , 'torn skirt' , 'thong leotard' , 'two-tone gloves' , 'red belt' , 'heart choker' , 'orange ribbon' , 'black bag' , 'kissing cheek' , 'scar on arm' , 'see-through silhouette' , 'heart-shaped eyewear' , 'policewoman' , 'ribbon-trimmed legwear' , 'hooded capelet' , 'blue belt' , 'bat print' , 'hair beads' , 'demon slayer uniform' , 'striped jacket' , 'pink shorts' , 'holding scythe' , 'holding pom poms' , 'grey sailor collar' , 'purple vest' , 'improvised gag' , 'reverse bunnysuit' , 'dangle earrings' , 'food print' , 'bike shorts under skirt' , 'print bow' , 'yellow scarf' , 'tight pants' , 'kiseru' , 'sleeves pushed up' , 'grabbing another\'s ass' , 'partially undressed' , 'blue scrunchie' , 'leotard aside' , 'bikini tan' , 'opaque glasses' , 'holding axe' , 'bandaids on nipples' , 'red moon' , 'anus peek' , 'idol clothes' , 'orange necktie' , 'pink leotard' , 'multiple belts' , 'sideless outfit' , 'flip-flops' , 'wizard hat' , 'holding pokemon' , 'aran sweater' , 'goth fashion' , 'single detached sleeve' , 'scar on nose' , 'plaid dress' , 'black armor' , 'asymmetrical footwear' , 'red sash' , 'vertical-striped dress' , 'leopard print' , 'rose print' , 'costume' , 'breast curtains' , 'unworn bra' , 'purple bodysuit' , 'ribbed shirt' , 'blue pantyhose' , 'pumps' , 'two-tone skirt' , 'swimsuit aside' , 'green cape' , 'spread toes' , 'maid bikini' , 'vegetable' , 'holding dagger' , 'bone hair ornament' , 'tail through clothes' , 'green leotard' , 'adjusting swimsuit' , 'green scarf' , 'furisode' , 'presenting armpit' , 'striped tail' , 'jacket around waist' , 'green sweater' , 'multiple crossover' , 'beer can' , 'gloved handjob' , 'medical eyepatch' , 'purple choker' , 'swim trunks' , 'holding underwear' , 'obiage' , 'frilled one-piece swimsuit' , 'coattails' , 'non-humanoid robot' , 'orange footwear' , 'yellow kimono' , 'multicolored wings' , 'pantylines' , 'blue hakama' , 'rectangular eyewear' , 'skirt suit' , 'aqua bowtie' , 'feather boa' , 'o-ring bottom' , 'purple wings' , 'american flag legwear' , 'letterman jacket' , 'single kneehigh' , 'ear ribbon' , 'barbell piercing' , 'black headband' , 'green coat' , 'mole on thigh' , 'multiple earrings' , 'domino mask' , 'layered clothes' , 'purple pants' , 'sake bottle' , 'grey panties' , 'kanzashi' , 'egyptian' , 'white wrist cuffs' , 'frilled kimono' , 'eyepatch bikini' , 'fur-trimmed hood' , 'bat hair ornament' , 'goggles on headwear' , 'burn scar' , 'open dress' , 'white scrunchie' , 'capri pants' , 'bra peek' , 'button badge' , 'rudder footwear' , 'red eyeliner' , 'holding drink' , 'wiffle gag' , 'black scrunchie' , 'white headband' , 'blue-framed eyewear' , 'nightcap' , 'bird on head' , 'american flag dress' , 'layered bikini' , 'diamond button' , 'print bowtie' , 'black camisole' , 'trefoil' , 'black cardigan' , 'horn bow' , 'naked sweater' , 'pink apron' , 'okobo' , 'gas mask' , 'green hairband' , 'two-tone swimsuit' , 'yoga pants' , 'yellow cardigan' , 'black robe' , 'hand over own mouth' , 'basketball \(object\)' , 'green bra' , 'hagoromo' , 'holding smoking pipe' , 'carrying person' , 'tie clip' , 'chef hat' , 'santa dress' , 'high-waist shorts' , 'green thighhighs' , 'orange gloves' , 'black mask' , 'heart tattoo' , 'rabbit hood' , 'four-leaf clover' , 'tunic' , 'rabbit print' , 'purple sleeves' , 'spider web print' , 'ankle socks' , 'grey socks' , 'sleepwear' , 'qingdai guanmao' , 'slime girl' , 'shark hair ornament' , 'holding flag' , 'ribbed dress' , 'gold earrings' , 'two-tone bikini' , 'lace panties' , 'tassel hair ornament' , 'chained' , 'wide spread legs' , 'loose belt' , 'food on head' , 'mandarin collar' , 'sailor bikini' , 'striped bra' , 'orange hairband' , 'ankle ribbon' , 'flower earrings' , 'strappy heels' , 'torn sleeves' , 'plunging neckline' , 'flats' , 'red pantyhose' , 'blue serafuku' , 'torn bodysuit' , 'crescent earrings' , 'purple umbrella' , 'harem outfit' , 'pink scrunchie' , 'hands in opposite sleeves' , 'strap between breasts' , 'holding arrow' , 'shirt tug' , 'meiji schoolgirl uniform' , 'purple cape' , 'wa maid' , 'undersized clothes' , 'orange bodysuit' , 'paw shoes' , 'cross scar' , 'suspender shorts' , 'see-through dress' , 'eye mask' , 'female pov' , 'holding controller' , 'gold chain' , 'microdress' , 'blue cardigan' , 'porkpie hat' , 'clitoral hood' , 'white sports bra' , 'thumb ring' , 'police hat' , 'holding doll' , 'white cardigan' , 'over-rim eyewear' , 'diagonal-striped bow' , 'heart-shaped pillow' , 'hanfu' , 'hugging doll' , 'unworn helmet' , 'multi-strapped bikini bottom' , 'multicolored swimsuit' , 'snowflake hair ornament' , 'purple shorts' , 'covered abs' , 'blue overalls' , 'snowflake print' , 'rubber boots' , 'checkered scarf' , 'holding sheath' , 'g-string' , 'breastless clothes' , 'black apron' , 'beard stubble' , 'purple scarf' , 'purple coat' , 'american flag bikini' , 'striped pants' , 'jirai kei' , 'black corset' , 'yellow scrunchie' , 'brown ribbon' , 'jackal ears' , 'heart in mouth' , 'side cutout' , 'bandaged head' , 'holding basket' , 'mismatched bikini' , 'red socks' , 'neck ruff' , 'feather trim' , 'pirate' , 'holding panties' , 'o-ring choker' , 'clover hair ornament' , 'evening gown' , 'scar on forehead' , 'pink bodysuit' , 'holding leash' , 'alternate legwear' , 'frilled ribbon' , 'orange shorts' , 'broken glass' , 'holding hammer' , 'unworn shirt' , 'sitting on desk' , 'enpera' , 'two-footed footjob' , 'torn cape' , 'holding condom' , 'keyhole' , 'bow bikini' , 'fashion' , 'orange kimono' , 'red armband' , 'black hakama' , 'holding sign' , 'shimakaze \(kancolle\) \(cosplay\)' , 'pink hoodie' , 'ice cube' , 'asticassia school uniform' , 'puffy detached sleeves' , 'aqua dress' , 'egyptian clothes' , 'uneven gloves' , 'medium dress' , 'fur coat' , 'looking over eyewear' , 'bodysuit under clothes' , 'open shorts' , 'skirt tug' , 'tailcoat' , 'pink-framed eyewear' , 'blue apron' , 'chest belt' , 'hadanugi dousa' , 'lace-trimmed dress' , 'feather earrings' , 'holding sack' , 'raincoat' , 'santa bikini' , 'brown sweater vest' , 'large hat' , 'yugake' , 'holding cat' , 'pith helmet' , 'brown capelet' , 'open-chest sweater' , 'pom pom hair ornament' , 'arabian clothes' , 'seamed legwear' , 'hand in panties' , 'grey bow' , 'chemise' , 'aqua skirt' , 'pant suit' , 'hooded cape' , 'wrestling outfit' , 'holding paintbrush' , 'red umbrella' , 'fur scarf' , 'pink sleeves' , 'sweets' , 'zero suit' , 'bomber jacket' , 'brown kimono' , 'rice bowl' , 'purple sweater' , 'high-waist pants' , 'tuxedo' , 'overskirt' , 'chain leash' , 'orange scrunchie' , 'negligee' , 'green choker' , 'borrowed clothes' , 'dessert' , 'diagonal-striped necktie' , 'brown sailor collar' , 'black leggings' , 'aqua ribbon' , 'large tail' , 'ainu clothes' , 'yellow vest' , 'sleeping upright' , 'red hoodie' , 'white suit' , 'pink pants' , 'carrot necklace' , 'character hair ornament' , 'humanoid robot' , 'grey-framed eyewear' , 'ribbon-trimmed skirt' , 'see-through legwear' , 'uneven sleeves' , 'belly chain' , 'white male underwear' , 'turtleneck dress' , 'ankle cuffs' , 'untied panties' , 'aqua bikini' , 'holding handheld game console' , 'striped sleeves' , 'cat hat' , 'aqua shirt' , 'red bag' , 'black sash' , 'frilled capelet' , 'shako cap' , 'diadem' , 'impossible bodysuit' , 'orange choker' , 'boxers' , 'holding clipboard' , 'brown cape' , 'holding syringe' , 'torn shorts' , 'mole on neck' , 'very long sleeves' , 'turban' , 'transparent umbrella' , 'skirt around one leg' , 'butterfly print' , 'naked jacket' , 'red one-piece swimsuit' , 'oni mask' , 'denim skirt' , 'spandex' , 'ribbed legwear' , 'notched lapels' , 'fanny pack' , 'tam o\' shanter' , 'green hoodie' , 'white sash' , 'magatama necklace' , 'star choker' , 'oversized animal' , 'egg \(food\)' , 'single pauldron' , 'purple sailor collar' , 'diving mask' , 'crotchless panties' , 'strapless bra' , 'vertical-striped skirt' , 'orange-tinted eyewear' , 'clothed animal' , 'clothes down' , 'yellow-framed eyewear' , 'legwear garter' , 'feather-trimmed sleeves' , 'traditional nun' , 'naked coat' , 'plaid bikini' , 'studded belt' , 'starry sky print' , 'shared scarf' , 'pendant choker' , 'impossible leotard' , 'korean clothes' , 'nontraditional playboy bunny' , 'holding pencil' , 'shuuchiin academy school uniform' , 'thigh ribbon' , 'fur-trimmed legwear' , 'oversized shirt' , 'holding lantern' , 'two-sided cape' , 'assault visor' , 'open skirt' , 'shark hood' , 'boxing gloves' , 'plaid bowtie' , 'glowing sword' , 'holding stick' , 'panty straps' , 'white-framed eyewear' , 'briefs' , 'multicolored bodysuit' , 'turnaround' , 'black wristband' , 'd-pad hair ornament' , 'champion\'s tunic \(zelda\)' , 'sweatband' , 'latex bodysuit' , 'yellow choker' , 'skull mask' , 'grey kimono' , 'vertical-striped pantyhose' , 'little busters! school uniform' , 'blood stain' , 'school hat' , 'gown' , 'sweater around waist' , 'red armor' , 'sleeping on person' , 'lace bra' , 'tactical clothes' , 'impossible dress' , 'standing on liquid' , 'hands on headwear' , 'u u' , 'spiked armlet' , 'holding whip' , 'millennium cheerleader outfit \(blue archive\)' , 'see-through leotard' , 'green neckerchief' , 'adjusting gloves' , 'o-ring thigh strap' , 'frilled socks' , 'swim briefs' , 'grey necktie' , 'puffy shorts' , 'cherry blossom print' , 'helm' , 'blue sash' , 'two-sided jacket' , 'plaid necktie' , 'kappougi' , 'multicolored legwear' , 'orange scarf' , 'backless leotard' , 'multicolored gloves' , 'holding swim ring' , 'multiple condoms' , 'loose clothes' , 'horned helmet' , 'mechanical tail' , 'stomach cutout' , 'naked sheet' , 'skull print' , 'chaps' , 'bird on hand' , 'black male underwear' , 'happy tears' , 'constellation print' , 'leg belt' , 'see-through skirt' , 'sports bikini' , 'blue eyeshadow' , 'handsfree ejaculation' , 'star necklace' , 'fine fabric emphasis' , 'open pants' , 'fishnet top' , 'drop earrings' , 'multicolored bikini' , 'barcode tattoo' , 'sleeve garter' , 'heart o-ring' , 'pink vest' , 'two-tone ribbon' , 'bear hair ornament' , 'chest strap' , 'bowl hat' , 'tight shirt' , 'brown necktie' , 'pencil dress' , 'gold hairband' , 'shared umbrella' , 'yellow hoodie' , 'condom packet strip' , 'white bag' , 'red-tinted eyewear' , 'animal ear headwear' , 'collared coat' , 'budget sarashi' , 'collared cape' , 'grey scarf' , 'grey border' , 'yellow thighhighs' , 'cloud print' , 'green sleeves' , 'yellow bra' , 'fish hair ornament' , 'poncho' , 'dixie cup hat' , 'leopard ears' , 'tankini' , 'bondage outfit' , 'scar on neck' , 'checkered kimono' , 'clover print' , 'ushanka' , 'lycoris uniform' , 'holding game controller' , 'untucked shirt' , 'pink socks' , 'black buruma' , 'winged helmet' , 'skull earrings' , 'side-tie leotard' , 'party hat' , 'green apron' , 'gusset' , 'gold necklace' , 'mouth veil' , 'polka dot dress' , 'puffy pants' , 'plaid headwear' , 'space helmet' , 'brown hairband' , 'blood from eyes' , 'sidepec' , 'strawberry print' , 'leather belt' , 'choko \(cup\)' , 'claw ring' , 'sanshoku dango' , 'super robot' , 'frilled cuffs' , 'two-tone bowtie' , 'pink blood' , 'single gauntlet' , 'taut dress' , 'holding brush' , 'checkered necktie' , 'tangzhuang' , 'cropped vest' , 'white serafuku' , 'fur-trimmed cloak' , 'straitjacket' , 'electrokinesis' , 'blue headband' , 'gold bracelet' , 'pink coat' , 'black undershirt' , 'stiletto heels' , 'polka dot bra' , 'holding baseball bat' , 'eyewear' , 'fertilization' , 'naked cape' , 'orange thighhighs' , 'thigh cutout' , 'ankle lace-up' , 'open bra' , 'ribbon-trimmed clothes' , 'polo shirt' , 'purple belt' , 'red bandana' , 'blue collar' , 'white veil' , 'belt bra' , 'yellow armband' , 'surprise kiss' , 'holding leaf' , 'german clothes' , 'fur-trimmed skirt' , 'shoulder boards' , 'flame print' , 'cupless bra' , 'holding shoes' , 'hooded sweater' , 'arm wrap' , 'multicolored shirt' , 'pillbox hat' , 'brown socks' , 'single fingerless glove' , 'plaid pants' , 'holding helmet' , 'yellow belt' , 'pink sailor collar' , 'red hood' , 'grey sleeves' , 'pocky kiss' , 'unworn bikini top' , 'striped gloves' , 'bow earrings' , 'fur-trimmed kimono' , 'cropped hoodie' , 'bandaid on hand' , 'biker clothes' , 'pink pajamas' , 'green-framed eyewear' , 'bandaged neck' , 'striped kimono' , 'crescent facial mark' , 'stitched face' , 'sweatpants' , 'shoulder strap' , 'sitting on stairs' , 'eyewear on headwear' , 'cowboy western' , 'pink collar' , 'grabbing another\'s chin' , 'respirator' , 'unworn boots' , 'ribbon bondage' , 'male playboy bunny' , 'thigh belt' , 'shoelaces' , 'purple capelet' , 'yellow bag' , 'bodice' , 'pink eyeshadow' , 'holding pillow' , 'dress tug' , 'pink belt' , 'reindeer costume' , 'ribbon-trimmed collar' , 'hakama pants' , 'snap-fit buckle' , 'pink one-piece swimsuit' , 'gold armor' , 'magatama earrings' , 'holding balloon' , 'brown-framed eyewear' , 'blue-tinted eyewear' , 'buttoned cuffs' , 'cow boy' , 'micro panties' , 'viewer holding leash' , 'wiping tears' , 'priest' , 'purple eyeshadow' , 'yellow sash' , 'holding scissors' , 'flip phone' , 'sitting on object' , 'multicolored bow' , 'romper' , 'diamond cutout' , 'large testicles' , 'single strap' , 'kissing forehead' , 'gold bikini' , 'grey belt' , 'black garter straps' , 'undone necktie' , 'orange sailor collar' , 'ankle strap' , 'holding needle' , 'triangle earrings' , 'bow choker' , 'striped shorts' , 'platform heels' , 'ribbed sleeves' , 'animal hug' , 'dress flower' , 'embellished costume' , 'hooded robe' , 'purple-tinted eyewear' , 'yellow pants' , 'heart button' , 'sunflower hair ornament' , 'hawaiian shirt' , 'bruise on face' , 'sleeveless turtleneck leotard' , 'plaid jacket' , 'lace-trimmed sleeves' , 'lizard tail' , 'orange neckerchief' , 'pointless condom' , 'drinking straw in mouth' , 'diving suit' , 'dirndl' , 'holding water gun' , 'two-tone headwear' , 'brown bowtie' , 'ribbon in mouth' , 'frilled shorts' , 'green bodysuit' , 'tricorne' , 'handkerchief' , 'spiked club' , 'cloth gag' , 'harem pants' , 'naked kimono' , 'vibrator under panties' , 'leather gloves' , 'sleeveless hoodie' , 'naked hoodie' , 'multicolored coat' , 'tribal' , 'colored shoe soles' , 'bow legwear' , 'sparkler' , 'mustache stubble' , 'greco-roman clothes' , 'butterfly on hand' , 'turtleneck leotard' , 'gradient clothes' , 'sleep mask' , 'hakurei reimu \(cosplay\)' , 'ass cutout' , 'latex gloves' , 'bath yukata' , 'santa boots' , 'bear print' , 'gold choker' , 'open robe' , 'drawing bow' , 'icho private high school uniform' , 'ginkgo leaf' , 'scar on stomach' , 'loose bowtie' , 'grey bikini' , 'unworn sandals' , 'yellow coat' , 'white armor' , 'forked tongue' , 'eyewear strap' , 'print bra' , 'pentacle' , 'shimaidon \(sex\)' , 'blue armor' , 'pink pantyhose' , 'kigurumi' , 'happi' , 'duffel coat' , 'pants rolled up' , 'unworn gloves' , 'short jumpsuit' , 'grey ribbon' , 'volleyball \(object\)' , 'deerstalker' , 'red apron' , 'star facial mark' , 'broken chain' , 'grey sports bra' , 'orange pants' , 'tulip hat' , 'untying' , 'orange pantyhose' , 'ajirogasa' , 'wrist guards' , 'grey bra' , 'ballerina' , 'full-length zipper' , 'novel cover' , 'cross print' , 'masturbation through clothes' , 'black garter belt' , 'purple one-piece swimsuit' , 'green capelet' , 'holding fishing rod' , 'two-tone footwear' , 'overcoat' , 'key necklace' , 'winged footwear' , 'brown apron' , 'high kick' , 'pink-tinted eyewear' , 'holding cane' , 'crescent print' , 'mask around neck' , 'brown hoodie' , 'print jacket' , 'jaguar ears' , 'lace-trimmed skirt' , 'open belt' , 'fishnet gloves' , 'naked bandage' , 'back-seamed legwear' , 'cocktail dress' , 'two-tone bodysuit' , 'brown bikini' , 'torn jeans' , 'holding vegetable' , 'purple hoodie' , 'sunflower field' , 'animal ear legwear' , 'holding hose' , 'new school swimsuit' , 'sphere earrings' , 'hamaya' , 'low neckline' , 'yellow apron' , 'green bag' , 'hatsune miku \(cosplay\)' , 'ribbed bodysuit' , 'impossible swimsuit' , 'triangle print' , 'sunscreen' , 'boxer briefs' , 'striped sweater' , 'candy hair ornament' , 'kesa' , 'gradient legwear' , 'holding jacket' , 'mismatched sleeves' , 'scooter' , 'kimono skirt' , 'orange ascot' , 'tooth necklace' , 'purple neckerchief' , 'double fox shadow puppet' , 'aqua panties' , 'sideless shirt' , 'leather boots' , 'goatee stubble' , 'pers' , 'camouflage jacket' , 'combat helmet' , 'grey neckerchief' , 'tapir tail' , 'single horizontal stripe' , 'white bird' , 'glomp' , 'diving mask on head' , 'gradient dress' , 'pointy footwear' , 'blood on knife' , 'torn scarf' , 'kouhaku nawa' , 'spiked choker' , 'sword on back' , 'pointing at another' , 'holding stylus' , 'scar on leg' , 'huge bow' , 'nippleless clothes' , 'aqua jacket' , 'circle skirt' , 'sleeve bow' , 'pearl bracelet' , 'orange hoodie' , 'hooded cardigan' , 'pink capelet' , 'yellow bodysuit' , 'two-tone sports bra' , 'combat boots' , 'rabbit-shaped pupils' , 'yin yang orb' , 'dice hair ornament' , 'fish print' , 'polka dot swimsuit' , 'ninja mask' , 'overall shorts' , 'holding ladle' , 'sweaty clothes' , 'shorts under dress' , 'fur-trimmed footwear' , 'shiny legwear' , 'drum set' , 'eden academy school uniform' , 'eyewear hang' , 'star brooch' , 'kirin \(armor\)' , 'expressive clothes' , 'burnt clothes' , 'ribbon-trimmed dress' , 'multicolored headwear' , 'duster' , 'wetsuit' , 'cross tie' , 'belt boots' , 'sharp toenails' , 'camouflage pants' , 'ribbed leotard' , 'torn leotard' , 'pinching sleeves' , 'strawberry hair ornament' , 'food-themed earrings' , 'white umbrella' , 'holding ice cream' , 'torn panties' , 'green socks' , 'clothes' , 'plaid panties' , 'mole above mouth' , 'riding pokemon' , 'athletic leotard' , 'headlamp' , 'sword behind back' , 'grey bodysuit' , 'fur-trimmed shorts' , 'frilled leotard' , 'jingasa' , 'brown corset' , 'bird mask' , 'orange panties' , 'hat tip' , 'tarot \(medium\)' , 'denim jacket' , 'two-tone hairband' , 'brown panties' , 'holding gohei' , 'white snake' , 'polka dot headwear' , 'white garter straps' , 'frilled ascot' , 'colored shadow' , 'yellow sleeves' , 'age regression' , 'shark costume' , 'cutout above navel' , 'purple scrunchie' , 'torn gloves' , 'two-tone legwear' , 'motorcycle helmet' , 'high-waist pantyhose' , 'mummy costume' , 'orange sweater' , 'mahjong tile' , 'unitard' , 'torn jacket' , 'bikesuit' , 'upshorts' , 'papakha' , 'lace-trimmed gloves' , 'silver trim' , 'scarf over mouth' , 'lace choker' , 'collared vest' , 'tented shirt' , 'ghost costume' , 'animal on lap' , 'ballet' , 'crystal earrings' , 'double w' , 'bicorne' , 'holding saucer' , 'multicolored footwear' , 'kourindou tengu costume' , 'red border' , 'pink border' , 'detective' , 'multicolored kimono' , 'drawing sword' , 'vampire costume' , 'shell bikini' , 'brown leotard' , 'pink ascot' , 'breast cutout' , 'two-tone leotard' , 'holding violin' , 'stole' , 'cetacean tail' , 'holding envelope' , 'sparkle print' , 'yellow leotard' , 'frog print' , 'yellow butterfly' , 'pink camisole' , 'panties on head' , 'lapel pin' , 'loungewear' , 'nearly naked apron' , 'long tail' , 'green hakama' , 'santa gloves' , 'kodona' , 'pearl earrings' , 'blue border' , 'boobplate' , 'heart collar' , 'training bra' , 'arm armor' , 'purple socks' , 'white mask' , 'fourth east high school uniform' , 'polka dot legwear' , 'uchikake' , 'print headwear' , 'pouring onto self' , 'egg hair ornament' , 'kamiyama high school uniform \(hyouka\)' , 'baggy clothes' , 'kine' , 'yellow cape' , 'native american' , 'hanten \(clothes\)' , 'holding bucket' , 'adjusting legwear' , 'lace gloves' , 'side drill' , 'sideburns stubble' , 'tube dress' , 'blue sports bra' , 'scar on mouth' , 'nejiri hachimaki' , 'gathers' , 'rook \(chess\)' , 'glowing butterfly' , 'thighhighs over pantyhose' , 'wakizashi' , 'swim cap' , 'fur cape' , 'grey capelet' , 'stained panties' , 'aviator sunglasses' , 'pill earrings' , 'blue robe' , 'prison clothes' , 'aqua footwear' , 'drying hair' , 'unzipping' , 'pinstripe shirt' , 'hat over one eye' , 'full beard' , 'bishop \(chess\)' , 'bandaid hair ornament' , 'huge moon' , 'hanbok' , 'loose shirt' , 'footwear ribbon' , 'tearing clothes' , 'white butterfly' , 'grey hairband' , 'ornate ring' , 'coin hair ornament' , 'holding tablet pc' , 'bucket hat' , 'gold footwear' , 'tutu' , 'holding popsicle' , 'between pectorals' , 'orange vest' , 'alpaca ears' , 'holding ribbon' , 'floating scarf' , 'mole on cheek' , 'crotch cutout' , 'single epaulette' , 'heart facial mark' , 'cropped sweater' , 'messenger bag' , 'weasel ears' , 'cowboy boots' , 'wrist bow' , 'upshirt' , 'in cup' , 'brown sleeves' , 'clothes between breasts' , 'swimsuit cover-up' , 'double vertical stripe' , 'kissing hand' , 'armpit cutout' , 'white hood' , 'brown choker' , 'chin strap' , 'gladiator sandals' , 'mole on stomach' , 'single boot' , 'red tank top' , 'black umbrella' , 'blue tunic' , 'wrist wrap' , 'single wrist cuff' , 'kepi' , 'white headdress' , 'wet dress' , 'hooded track jacket' , 'orange sleeves' , 'brown collar' , 'two-tone cape' , 'hooded bodysuit' , 'red mask' , 'body armor' , 'red mittens' , 'torn swimsuit' , 'purple sash' , 'satin' , 'alice \(alice in wonderland\) \(cosplay\)' , 'cat ear legwear' , 'saiyan armor' , 'white mittens' , 'grey cape' , 'frilled sailor collar' , 'side slit shorts' , 'pants tucked in' , 'condom belt' , 'cross choker' , 'black sweater vest' , 'rider belt' , 'multicolored cape' , 'yellow socks' , 'fold-over boots' , 'pink hakama' , 'naked overalls' , 'spit take' , 'leg wrap' , 'mochi trail' , 'sleeve ribbon' , 'blood on arm' , 'tied jacket' , 'blue tank top' , 'two-sided dress' , 'holding beachball' , 'clothes between thighs' , 'purple-framed eyewear' , 'jockstrap' , 'lowleg pants' , 'flying kick' , 'tight dress' , 'holding jewelry' , 'frilled camisole' , 'unworn coat' , 'see-through jacket' , 'pink cape' , 'sideways hat' , 'holding megaphone' , 'string bra' , 'huge testicles' , 'unworn dress' , 'holding letter' , 'coke-bottle glasses' , 'open bodysuit' , 'holding behind back' , 'holding chocolate' , 'studded bracelet' , 'aqua gloves' , 'star pasties' , 'multicolored scarf' , 'test plugsuit' , 'levitation' , 'houndstooth' , 'yellow tank top' , 'polka dot skirt' , 'chalice' , 'vertical-striped jacket' , 'leather pants' , 'hard hat' , 'cardigan around waist' , 'vertical-striped bikini' , 'torn bodystocking' , 'purple ascot' , 'white tiger' , 'arachne' , 'cross pasties' , 'holding money' , 'two-tone hoodie' , 'latex legwear' , 'grey tank top' , 'back-print panties' , 'sitting on rock' , 'breasts on table' , 'green pantyhose' , 'heart maebari' , 'male maid' , 'arm cuffs' , 'floral print kimono' , 'fake nails' , 'ribbon-trimmed headwear' , 'clown' , 'jaguar print' , 'adjusting necktie' , 'jeweled branch of hourai' , 'multi-strapped panties' , 'holding notebook' , 'pool of blood' , 'yellow raincoat' , 'flower over eye' , 'cardigan vest' , 'bridal legwear' , 'yellow-tinted eyewear' , 'striped hoodie' , 'naked scarf' , 'dudou' , 'green tunic' , 'otter ears' , 'purple hakama' , 'green tank top' , 'hand under shirt' , 'skirt basket' , 'white romper' , 'sitting backwards' , 'youtou high school uniform' , 'vietnamese dress' , 'lace-trimmed hairband' , 'bear costume' , 'green belt' , 'crotchless pantyhose' , 'wringing clothes' , 'holding branch' , 'shorts around one leg' , 'aqua neckerchief' , 'holding remote control' , 'nose ring' , 'polka dot shirt' , 'underbutt' , 'holding skull' , 'four-leaf clover hair ornament' , 'potara earrings' , 'grey leotard' , 'print necktie' , 'parka' , 'shell necklace' , 'holding sex toy' , 'blue bandana' , 'long beard' , 'orange belt' , 'pers' , 'camouflage headwear' , 'penguin hood' , 'crocs' , 'jacket over swimsuit' , 'rope belt' , 'polar bear' , 'shoulder sash' , 'fur boots' , 'checkered sash' , 'yellow sweater vest' , 'purple cardigan' , 'anchor necklace' , 'striped hairband' , 'brown bra' , 'sailor senshi' , 'bike shorts under shorts' , 'hat with ears' , 'puff and slash sleeves' , 'stitched mouth' , 'half mask' , 'print sleeves' , 'green scrunchie' , 'thick mustache' , 'argyle sweater' , 'hospital gown' , 'onesie' , 'green armband' , 'polka dot scrunchie' , 'double \\m/' , 'two-tone coat' , 'cherry hair ornament' , 'sukajan' , 'platform boots' , 'floating weapon' , 'wa lolita' , 'striped one-piece swimsuit' , 'cat costume' , 'jaguar tail' , 'rabbit hat' , 'thick beard' , 'yellow border' , 'martial arts belt' , 'bib' , 'fur-trimmed collar' , 'surcoat' , 'single thigh boot' , 'strawberry panties' , 'high tops' , 'sitting on table' , 'plaid bra' , 'blood bag' , 'ankle wrap' , 'print hoodie' , 'green-tinted eyewear' , 'dress swimsuit' , 'flower brooch' , 'cross-laced legwear' , 'popped button' , 'blue shawl' , 'butterfly brooch' , 'white sarong' , 'green one-piece swimsuit' , 'grey serafuku' , 'lace-trimmed thighhighs' , 'orange cape' , 'american flag print' , 'skirt flip' , 'ehoumaki' , 'chain headband' , 'holding frying pan' , 'orange leotard' , 'sling bikini top' , 'adapted uniform' , 'kabuto \(helmet\)' , 'planet hair ornament' , 'hair color connection' , 'patchwork clothes' , 'hat on back' , 'watermelon slice' , 'holding teapot' , 'pants under skirt' , 'unworn bikini bottom' , 'popsicle in mouth' , 'milky way' , 'multicolored hairband' , 'drop-shaped pupils' , 'skull necklace' , 'purple serafuku' , 'mitre' , 'frilled jacket' , 'aqua bra' , 'blue pajamas' , 'anchor choker' , 'polka dot ribbon' , 'halter shirt' , 'red sports bra' , 'nudist' , 'naked tabard' , 'sideless kimono' , 'single knee pad' , 'long shirt' , 'multiple scars' , 'cross-laced slit' , 'card parody' , 'orange socks' , 'cream on face' , 'sam browne belt' , 'satin panties' , 'embroidery' , 'blue sarong' , 'pink umbrella' , 'buruma aside' , 'genderswap \(otf\)' , 'blue umbrella' , 'legband' , 'musical note print' , 'holding wrench' , 'unworn eyepatch' , 'hooded dress' , 'floating book' , 'rabbit costume' , 'skeleton print' , 'wataboushi' , 'st\. theresa\'s girls academy school uniform' , 'pinstripe suit' , 'bowler hat' , 'pegasus knight uniform \(fire emblem\)' , 'green eyeshadow' , 'pumpkin hair ornament' , 'bandaged wrist' , 'holding swimsuit' , 'spiked hairband' , 'coat dress' , 'jester' , 'stopwatch' , 'shoulder belt' , 'holding footwear' , 'holding toy' , 'panties under buruma' , 'food art' , 'hugging book' , 'brown border' , 'half-skirt' , 'orange jumpsuit' , 'midriff sarashi' , 'red track suit' , 'grey suit' , 'hooded vest' , 'scylla' , 'bathrobe' , 'coif' , 'bikini shorts' , 'bow skirt' , 'side-tie peek' , 'bralines' , 'blue camisole' , 'striped coat' , 'pelt' , 'unfastened' , 'greek toe' , 'black armband' , 'adjusting panties' , 'vertical-striped socks' , 'plaid ribbon' , 'vertical-striped panties' , 'print sarong' , 'cloth' , 'holding test tube' , 'band uniform' , 'checkered shirt' , 'lowleg skirt' , 'fur-trimmed shirt' , 'german flag bikini' , 'lightning bolt print' , 'holding mop' , 'blue tabard' , 'holly hair ornament' , 'exercise ball' , 'lillian girls\' academy school uniform' , 'vertical-striped pants' , 'blood on leg' , 'stained clothes' , 'high-low skirt' , 'christmas stocking' , 'tengu mask' , 'pumpkin hat' , 'hand wraps' , 'belt skirt' , 'silver dress' , 'lace-trimmed choker' , 'brown mittens' , 'shiny and normal' , 'blue hood' , 'naked cloak' , 'one-piece thong' , 'black bandeau' , 'orange goggles' , 'fishnet socks' , 'purple collar' , 'flower choker' , 'elbow sleeve' , 'holding heart' , 'pocky in mouth' , 'grey apron' , 'jiangshi costume' , 'mizu happi' , 'rubber gloves' , 'red cardigan' , 'holding coin' , 'mole under each eye' , 'clothes theft' , 'holding microphone stand' , 'clock eyes' , 'holding chain' , 'wrong foot' , 'converse' , 'thong aside' , 'knight \(chess\)' , 'mutual hug' , 'brown neckerchief' , 'kerchief' , 'red suit' , 'red robe' , 'strapless bottom' , 'wing brooch' , 'diagonal-striped bowtie' , 'holding drumsticks' , 'aqua kimono' , 'vertical-striped kimono' , 'stitched arm' , 'pink sash' , 'cuff links' , 'checkered dress' , 'ornate border' , 'animal ear hairband' , 'grey bowtie' , 'toe cleavage' , 'yellow camisole' , 'crotch zipper' , 'shirt overhang' , 'animal on hand' , 'holding shirt' , 'unworn shorts' , 'riding bicycle' , 'star-shaped eyewear' , 'orange headband' , 'scouter' , 'long toenails' , 'holding cake' , 'cargo pants' , 'frilled umbrella' , 'glitter' , 'holding suitcase' , 'green headband' , 'micro bra' , 'motosu school uniform' , 'brown serafuku' , 'single head wing' , 'covered clitoris' , 'panda hood' , 'taut swimsuit' , 'purple butterfly' , 'aqua leotard' , 'little red riding hood \(grimm\) \(cosplay\)' , 'fur cuffs' , 'glowing hand' , 'panties under shorts' , 'maple leaf print' , 'exploding clothes' , 'holding creature' , 'stiletto \(weapon\)' , 'clawed gauntlets' , 'print mug' , 'frilled headwear' , 'cable tail' , 'red male underwear' , 'exposed pocket' , 'two-sided coat' , 'safety glasses' , 'holding fish' , 'front slit' , 'flippers' , 'kariyushi shirt' , 'knives between fingers' , 'broken sword' , 'policeman' , 'spade hair ornament' , 'male underwear peek' , 'leotard peek' , 'neck garter' , 'weasel tail' , 'blue suit' , 'holding photo' , 'dissolving clothes' , 'holding pole' , 'holding shovel' , 'backless swimsuit' , 'tickling armpits' , 'low-cut armhole' , 'propeller hair ornament' , 'fake magazine cover' , 'holding cross' , 'otter tail' , 'taut leotard' , 'o-ring swimsuit' , 'wind turbine' , 'pom pom earrings' , 'checkered bow' , 'multiple hairpins' , 'studded choker' , 'red bandeau' , 'single garter strap' , 'fruit hat ornament' , 'ski goggles' , 'holding briefcase' , 'brown sash' , 'layered kimono' , 'o-ring belt' , 'striped vest' , 'green cardigan' , 'multicolored stripes' , 'aqua hairband' , 'plate carrier' , 'bear hood' , 'holding bra' , 'detached leggings' , 'paw print pattern' , 'multicolored tail' , 'walker \(robot\)' , 'down jacket' , 'rabbit on head' , 'holding scroll' , 'pink tank top' , 'yellow one-piece swimsuit' , 'white bandeau' , 'black tube top' , 'scoop neck' , 'temari ball' , 'red wine' , 'yellow pantyhose' , 'bandaged fingers' , 'ahoge wag' , 'black hood' , 'black veil' , 'head on head' , 'hakama shorts' , 'moose ears' , 'fishnet bodysuit' , 'pointy hat' , 'fur jacket' , 'bandaid on neck' , 'holding surfboard' , 'bridal lingerie' , 'hat belt' , 'overall skirt' , 'holding map' , 'knife sheath' , 'rotary phone' , 'pantyhose under swimsuit' , 'pawn \(chess\)' , 'unworn goggles' , 'sky lantern' , 'frontless outfit' , 'armored leotard' , 'shoulder plates' , 'ribbed thighhighs' , 'forked tail' , 'lightning bolt hair ornament' , 'undone neck ribbon' , 'shoulder guard' , 'lop rabbit ears' , 'cassock' , 'metamoran vest' , 'normal suit' , 'checkered legwear' , 'see-through swimsuit' , 'holding necklace' , 'panties over pantyhose' , 'orange bra' , 'adjusting scarf' , 'layered shirt' , 'bird on arm' , 'paint on clothes' , 'scar on hand' , 'blue outline' , 'unworn bikini' , 'pink sports bra' , 'tape on nipples' , 'adjusting buruma' , 'side-tie shirt' , 'torn coat' , 'rash guard' , 'poke ball \(legends\)' , 'ankle bow' , 'mtu virus' , 'bandaid on head' , 'fur-trimmed bikini' , 'hat tassel' , 'argyle cutout' , 'cross-laced skirt' , 'fruit on head' , 'cow costume' , 'multicolored leotard' , 'white garter belt' , 'holding toothbrush' , 'toga' , 'holding lipstick tube' , 'multi-strapped bikini top' , 'white wristband' , 'purple robe' , 'turtleneck jacket' , 'rice hat' , 'shared earphones' , 'mole on arm' , 'holding mirror' , 'corsage' , 'black outline' , 'anchor earrings' , 'wrapped candy' , 'gingham' , 'sweet lolita' , 'side-tie skirt' , 'print scarf' , 'green collar' , 'sweater tucked in' , 'front-print panties' , 'square neckline' , 'bear panties' , 'mini witch hat' , 'holding key' , 'holding torch' , 'holding plectrum' , 'white tube top' , 'unworn hair ornament' , 'holding magnifying glass' , 'single off shoulder' , 'torn cloak' , 'heart hair' , 'shirt around waist' , 'sailor swimsuit \(idolmaster\)' , 'detached ahoge' , 'ankle garter' , 'singlet' , 'aviator cap' , 'aqua shorts' , 'holding newspaper' , 'female service cap' , 'ankleband' , 'black babydoll' , 'multiple bracelets' , 'front zipper swimsuit' , 'kin-iro mosaic high school uniform' , 'holding bell' , 'blue male underwear' , 'side cape' , 'glove bow' , 'green serafuku' , 'claw foot bathtub' , 'ribbed socks' , 'dress shoes' , 'vertical-striped shorts' , 'blue sweater vest' , 'fur-trimmed thighhighs' , 'streetwear' , 'vertical stripes' , 'labcoat' , 'argyle' , 'print legwear' , 'tight' , 'legwear under shorts' , 'multi-strapped bikini' , 'diagonal stripes' , 'gothic' , 'frilled swimsuit' , 'bunny print' , 'qing guanmao' , 'matching outfit' , 'borrowed garments' , 'beltbra' , 'nike' , 'traditional clothes' , 'power suit \(metroid\)' , 'clog sandals' , 'multiple straps' , 'catholic' , 'strapless swimsuit' , 'sling' , 'bunny hat' , 'beltskirt' , 'greek clothes' , 'military helmet' , 'hardhat' , 'yamakasa'
}

not_scene_tags = appearance_tags.union(clothing_tags)

# 內建的黑名單
builtin_blacklist = [
    r'^.*from .*$', r'^.*focus.*$', r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', 
    r'^.*censor.*$', r'^.* name$', r'^signature$', r'^.* username$', r'^.*text.*$', 
    r'^.* bubble$', r'^multiple views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
    r'^traditional media$', r'^parody$', r'^.*cover$', r'^.* theme$', r'^realistic$', 
    r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
    r'^omake$', r'^column.*$', r'^.* (medium)$', r'^manga$', r'^lineart$', r'^.*logo$',  r'^greyscale$',
    r'^.*photorealistic.*$', r'^tegaki$', r'^sketch$', r'^silhouette$', r'^web address$', r'^.*border$',
    r'^(close up|dutch angle|downblouse|downpants|pantyshot|upskirt|atmospheric perspective|fisheye|panorama|perspective|pov|rotated|sideways|upside-down|vanishing point)$',
    r'^(face|cowboy shot|portrait|upper body|lower body|feet out of frame|full body|wide shot|very wide shot|close-up|cut-in|cropped legs|head out of frame|cropped torso|cropped arms|cropped shoulders|profile|group profile)$'
]

blacklist_patterns = [re.compile(pattern) for pattern in builtin_blacklist]

def run_openai_api(image_data, model=MODEL):

    def encode_image(image_buffer):
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    base64_image = encode_image(image_data)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Help me with my task! Use the simplest English answer, put the key answer in \"\". Only provide one answer."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Do most of these images (about >2/3) depict the same outfit? If yes, give a fitting creative English costume name for this outfit in \"\", using two-three words. If no, answer \"no\". If the outfit is similar to one previously named, use the same name. If not, try to avoid using any previously used words. Only provide one answer."},
            {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}
        ]}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    
    tags_text = response.choices[0].message.content

    return tags_text

def combine_images(images: List[str]) -> Optional[np.ndarray]:

    def imread_unicode(file_path: str) -> Optional[np.ndarray]:
        try:
            stream = open(file_path, "rb")
            bytes = bytearray(stream.read())
            numpy_array = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Warning: Unable to read image {file_path}. Error: {e}")
            return None

    def resize_to_height(image, height):
        ratio = height / image.shape[0]
        new_width = int(image.shape[1] * ratio)
        resized_image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
        return resized_image

    loaded_images = []

    for img_path in images:
        img = imread_unicode(img_path)
        if img is None:
            continue
        loaded_images.append(img)

    if not loaded_images:
        print("Error: No images loaded successfully.")
        return None

    if len(loaded_images) <= 4:
        # 單排排列圖像
        new_height = 512
        resized_images = [resize_to_height(img, new_height) for img in loaded_images]
        total_width = sum(img.shape[1] for img in resized_images)
        combined_image = np.zeros((new_height, total_width, 3), dtype=np.uint8)
        x_offset = 0
        for img in resized_images:
            combined_image[:, x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
    else:
        # 雙排排列圖像
        new_height = 256
        resized_images = [resize_to_height(img, new_height) for img in loaded_images]
        num_images = len(resized_images)
        num_rows = 2
        num_cols = (num_images + 1) // num_rows  # 每排圖像數

        row_heights = [sum(img.shape[1] for img in resized_images[i::num_rows]) for i in range(num_rows)]
        max_width = max(row_heights)

        combined_image = np.zeros((new_height * num_rows, max_width, 3), dtype=np.uint8)

        for row in range(num_rows):
            x_offset = 0
            y_offset = row * new_height
            for col in range(num_cols):
                idx = row + col * num_rows
                if idx < num_images:
                    img = resized_images[idx]
                    combined_image[y_offset:y_offset + new_height, x_offset:x_offset + img.shape[1]] = img
                    x_offset += img.shape[1]

    return combined_image

def manual_check(combined_image: np.ndarray, existing_costumes: List[str], cluster_name: str, cluster_prompt: str) -> str:

    def display_combined_image(image: np.ndarray, title: str):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    if combined_image is not None:
        print(combined_image)
        display_combined_image(combined_image, f"Manual Check - {cluster_name} - {cluster_prompt}")
    
    print(f"目前已有的服裝命名有: {', '.join(existing_costumes)}")
    costume_name = input(f"使用預設服裝名 {cluster_name} (按 ENTER 同意) 或輸入其他名稱，如果不是服裝回答n或no: ").strip().lower()
    if costume_name == 'n':
        return 'no'
    if not costume_name:
        costume_name = cluster_name

    return costume_name

def process_clustering(image_info_list: List[Dict[str, Optional[str]]], tags_list, n_clusters, cluster_prefix, args):

    def extract_text_features(tags_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
        X = vectorizer.fit_transform(tags_list).toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()
        return X, feature_names

    def perform_clustering(X: np.ndarray, n_clusters: int, model_name: str) -> np.ndarray:
        from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
        if model_name == "K-Means聚類":
            model = KMeans(n_clusters=n_clusters, n_init=8)
        elif model_name == "Spectral譜聚類":
            model = SpectralClustering(n_clusters=n_clusters, affinity='cosine')
        elif model_name == "Agglomerative層次聚類":
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        elif model_name == "OPTICS聚類":
            model = OPTICS(metric="cosine", min_samples=n_clusters)
        else:
            raise ValueError(f"不支持的聚類模型: {model_name}")
        return model.fit_predict(X)

    def cluster_feature_analysis(X: np.ndarray, y_pred: np.ndarray, feature_names: List[str], clusters_ID: np.ndarray) -> List[dict]:
        from sklearn.feature_selection import SelectKBest, chi2
        import pandas as pd
        cluster_feature_tags_list = []
        for i in tqdm(clusters_ID, desc="分析聚類特徵"):
            temp_pred = y_pred.copy()
            temp_pred[temp_pred != i] = i + 1
            k = min(10, X.shape[1])
            selector = SelectKBest(chi2, k=k)
            X_shifted = X - X.min(axis=0)
            X_selected = selector.fit_transform(X_shifted, temp_pred)
            X_selected_index = selector.get_support(indices=True)
            tags_selected = np.array(feature_names)[X_selected_index]
            cluster_df = pd.DataFrame(X_shifted[temp_pred == i], columns=feature_names)
            cluster_tags_df = cluster_df.iloc[:, X_selected_index]
            mean_values = cluster_tags_df.mean(axis=0)
            prompt_tags_list = mean_values.nlargest(10).index.tolist()
            cluster_feature_tags_list.append({"prompt": prompt_tags_list})
        return cluster_feature_tags_list

    def is_nsfw(tags: str) -> bool:
        """檢查標籤是否符合 NSFW 黑名單"""
        nsfw_blacklist = ['.*nude.*$', '.*penis.*$', '.*nipple.*$', '.*anus.*$', '.*sex.*$']
        for pattern in nsfw_blacklist:
            if re.search(pattern, tags, re.IGNORECASE):
                return True
        return False
        
    def is_clustering(tags: str, input_set: Set[str]) -> bool:
        tags_list = tags.split(', ')
        tags_set = set(tags_list)
        intersection = tags_set.intersection(input_set)
        if len(intersection) >= 2:
            return True
        return False

    def update_clusters(image_info_list: List[Dict[str, Optional[str]]], y_pred: np.ndarray, cluster_feature_tags_list: List[dict], cluster_prefix: str, naming_mode: str):
        cluster_counts = np.bincount(y_pred)
        non_empty_clusters = [i for i in range(len(cluster_counts)) if cluster_counts[i] > 0]
        sorted_clusters = sorted(non_empty_clusters, key=lambda x: cluster_counts[x], reverse=True)

        existing_costumes = []    
        for idx, cluster_id in enumerate(sorted_clusters):
            nsfw = False
            cluster_prompt = ', '.join(cluster_feature_tags_list[cluster_id]['prompt'])
            cluster_name = 'no'
            if cluster_prefix == "costume_" or cluster_prefix == "appearance_":
                if idx < 20 and is_clustering(cluster_prompt, clothing_tags if cluster_prefix == "costume_" else appearance_tags):
                    cluster_name = f"{cluster_prefix}{idx}" 
                    if naming_mode != 'auto':
                        images = [info['path'] for info in image_info_list if y_pred[image_info_list.index(info)] == cluster_id and not is_nsfw(info['all_tags'])][:8]
                        if images is None:
                            images = [info['path'] for info in image_info_list if y_pred[image_info_list.index(info)] == cluster_id][:8]
                            nsfw = True
                        combined_image = combine_images(images)
                    if naming_mode == 'manual':
                        cluster_name = manual_check(combined_image, existing_costumes, cluster_name, cluster_prompt)
                    if naming_mode == 'gpt4o':
                        if combined_image is not None:
                            pil_image = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
                            rating, _ = anime_dbrating(pil_image)  
                            if rating in ['general', 'sensitive', 'questionable'] and not nsfw:
                                
                                image_buffer = BytesIO()
                                Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)).save(image_buffer, format="WEBP")
                                image_buffer.seek(0)
                                gpt_response = run_openai_api(image_buffer).lower().replace('"','')
                                print(gpt_response)
                                cluster_name = gpt_response
                            else:
                                print("聚類是NSFW，將手動檢查")
                                cluster_name = manual_check(combined_image, existing_costumes, cluster_name, cluster_prompt)
                    if cluster_name and cluster_name != 'no':
                        existing_costumes.append(cluster_name)
            else:
                if idx < 20:
                    cluster_name = f"{cluster_prefix}{idx}"
            for image_info in image_info_list:
                if y_pred[image_info_list.index(image_info)] == cluster_id:
                    if cluster_name != 'no':
                        image_info[f'{cluster_prefix}cluster_name'] = cluster_name
                    image_info[f'{cluster_prefix}cluster_prompt'] = cluster_prompt
       
    X, feature_names = extract_text_features(tags_list)
    if len(tags_list) > 0:
        y_pred = perform_clustering(X, n_clusters, args.cluster_model_name)
        clusters_ID = np.unique(y_pred)
        cluster_feature_tags_list = cluster_feature_analysis(X, y_pred, feature_names, clusters_ID)

        update_clusters(image_info_list, y_pred, cluster_feature_tags_list, cluster_prefix, args.naming_mode)

def process_subfolder(subfolder_path: str, args, md_filepath: str):

    def read_images_and_tags(images_dir: str, file_ext: str = '.txt') -> List[Dict[str, Optional[str]]]:

        def whitelist_tags(tags: str, input_set: Set[str]) -> str:
            tags_list = tags.split(', ')
            tags_set = set(tags_list)
            intersection = tags_set.intersection(input_set)
            return ', '.join(intersection)

        def filter_tags(tags: str, input_set: Set[str]) -> str:
            tags_list = tags.split(', ')
            tags_set = set(tags_list)
            filtered_tags = tags_set - input_set
            return ', '.join(filtered_tags)

        def repeat_tags(tags: str, input_set: Set[str], repeat_count: int = 2) -> str:
            tags_list = tags.split(', ')
            tags_set = set(tags_list)
            intersection = tags_set.intersection(input_set)
            result_tags = tags_list
            for i in range(repeat_count):
                result_tags.extend(intersection)
            return ', '.join(result_tags)

        image_info_list = []
        image_base_names = set(os.path.splitext(file)[0] for ext in IMAGE_EXTENSIONS for file in glob.glob(os.path.join(images_dir, f"*{ext.lower()}")) + glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
        
        for base_name in image_base_names:
            txt_file = os.path.join(images_dir, f"{base_name}{file_ext}")
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                possible_image_path = os.path.join(images_dir, f"{base_name}{ext}")
                if os.path.exists(possible_image_path):
                    image_path = possible_image_path
                    break
            
            if image_path and os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if '|||' in first_line:
                        tags = first_line.split('|||')[1].strip() if len(first_line.split('|||')) > 1 else ''
                    else:
                        tags = first_line.split(', ', 1)[1] if ', ' in first_line else first_line
                    image_info_list.append({
                        'path': image_path,
                        'costume': repeat_tags(tags, not_scene_tags),
                        'appearance': repeat_tags(tags, appearance_tags, repeat_count=3),
                        'scene': filter_tags(tags, clothing_tags),
                        'all_tags': tags,
                        'costume_cluster_name': None,
                        'costume_cluster_prompt': None,
                        'appearance_cluster_name': None,
                        'appearance_cluster_prompt': None,
                        'scene_cluster_name': None,
                        'scene_cluster_prompt': None                 
                    })
        
        return image_info_list

    def insert_cluster_text_to_txt(info: Dict[str, Optional[str]], args):
        def contains_color(tag: str) -> bool:
            colors = {'red', 'orange', 'yellow', 'green', 'blue', 'aqua', 'purple', 'brown', 'pink', 'black', 'white', 'grey', 'dark-', 'light ', 'pale', 'blonde'}
            return any(color in tag for color in colors)

        txt_filepath = os.path.splitext(info['path'])[0] + '.txt'
        cluster_name = info.get('costume_cluster_name', '')

        # 檢查每個標籤是否為 None，並組合有效的標籤
        costume_cluster_prompt = info.get('costume_cluster_prompt', '')
        appearance_cluster_prompt = info.get('scene_appearance_prompt', '')
        scene_cluster_prompt = info.get('scene_cluster_prompt', '')

        # 分割標籤，過濾空標籤
        cluster_costume_tags = ', '.join(filter(None, [costume_cluster_prompt])).split(', ')
        cluster_appearance_tags = ', '.join(filter(None, [appearance_cluster_prompt])).split(', ')
        cluster_scene_tags = ', '.join(filter(None, [scene_cluster_prompt])).split(', ')

        with open(txt_filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            return  

        new_tags = info['all_tags'].split(', ')
        cluster_text = ''
        new_caption = ''
        if args.naming_mode != 'auto':
            cluster_text += f'{cluster_name}, '

        for tag in new_tags:
            if not contains_color(tag) and tag not in cluster_text and tag not in new_caption:
                if tag in cluster_costume_tags:
                    cluster_text += f'{tag}, '
                elif tag in cluster_scene_tags and ('explicit' in lines[0] or 'questionable' in lines[0]):
                    cluster_text += f'{tag}, '
                elif tag in cluster_appearance_tags and tag in appearance_tags:
                    cluster_text += f'{tag}, '
                else:
                    new_caption += f'{tag}, '

        for i in range(len(lines)):
            line = lines[i].strip()
            if new_caption:
                line = line.replace(info['all_tags'], new_caption)
            parts = line.split(', ', 1)
            if len(parts) > 1:
                lines[i] = f"{parts[0]}, {cluster_text}{parts[1]}"
            else:
                lines[i] = f"{line}, {cluster_text}"
            lines[i] = ', '.join(lines[i].split(', '))  # 合併標籤，保持格式一致     

        with open(txt_filepath, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))
        
    def copy_or_move_clusters(subfolder_path: str, subfolder_name: str, image_info_list: List[Dict[str, Optional[str]]], repeats: int, name_from_folder: str, args):
        cluster_images = {}
        for info in image_info_list:
            cluster_name = info[f'{args.dir_mode}_cluster_name'] 
            if cluster_name:
                if cluster_name not in cluster_images:
                    cluster_images[cluster_name] = []
                cluster_images[cluster_name].append(info['path'])

        # 計算每個聚類要複製幾份
        max_cluster_size = max(len(images) for images in cluster_images.values())
        num_subfolder_images = len(image_info_list) * repeats
        extra_repeats = max(1, int(math.ceil(num_subfolder_images / (max_cluster_size * len(cluster_images)))))

        for cluster_name, images in cluster_images.items():
            num_copies = min(15, int(max_cluster_size / len(images)))

            if args.copy_cluster:
                extra_folder_name = f"{extra_repeats}_{name_from_folder} extra hard link"
                extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                os.makedirs(extra_folder_path, exist_ok=True)

                for i in range(num_copies):
                    for img_path in images:
                        img_base_name = os.path.basename(img_path)
                        new_img_path = os.path.join(extra_folder_path, f"{i}_{img_base_name}")
                        try:
                            os.link(img_path, new_img_path)
                        except (FileExistsError, OSError) as e:
                            if isinstance(e, OSError):
                                print(f"硬連結失敗 {img_path} -> {new_img_path}: {e}, 將使用複製")
                                shutil.copy(img_path, new_img_path)
                                extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                os.makedirs(extra_folder_path, exist_ok=True)
                            else:
                                print(f"文件已存在: {new_img_path}")

                        txt_file = os.path.splitext(img_path)[0] + '.txt'
                        if os.path.exists(txt_file):
                            new_txt_path = os.path.join(extra_folder_path, f"{i}_{os.path.basename(txt_file)}")
                            try:
                                os.link(txt_file, new_txt_path)
                            except (FileExistsError, OSError) as e:
                                if isinstance(e, OSError):
                                    print(f"硬連結失敗 {txt_file} -> {new_txt_path}: {e}, 將使用複製")
                                    shutil.copy(txt_file, new_txt_path)
                                    extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                    extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                    os.makedirs(extra_folder_path, exist_ok=True)
                                else:
                                    print(f"文件已存在: {new_txt_path}")

                        npz_file = os.path.splitext(img_path)[0] + '.npz'
                        if os.path.exists(npz_file):
                            new_npz_path = os.path.join(extra_folder_path, f"{i}_{os.path.basename(npz_file)}")
                            try:
                                os.link(npz_file, new_npz_path)
                            except (FileExistsError, OSError) as e:
                                if isinstance(e, OSError):
                                    print(f"硬連結失敗 {npz_file} -> {new_npz_path}: {e}, 將使用複製")
                                    shutil.copy(npz_file, new_npz_path)
                                    extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                    extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                    os.makedirs(extra_folder_path, exist_ok=True)
                                else:
                                    print(f"文件已存在: {new_npz_path}")

            if args.move_cluster:
                cluster_dir = os.path.join(subfolder_path, f"{num_copies}_{cluster_name}")
                os.makedirs(cluster_dir, exist_ok=True)
                for img_path in images:
                    try:
                        shutil.move(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))
                        txt_file = os.path.splitext(img_path)[0] + '.txt'
                        if os.path.exists(txt_file):
                            shutil.move(txt_file, os.path.join(cluster_dir, os.path.basename(txt_file)))
                        npz_file = os.path.splitext(img_path)[0] + '.npz'
                        if os.path.exists(npz_file):
                            shutil.move(npz_file, os.path.join(cluster_dir, os.path.basename(npz_file)))
                    except FileNotFoundError as e:
                        print(f"文件未找到: {e.filename}")

        if not args.copy_cluster and args.move_cluster:
            root_dir = os.path.join(subfolder_path, '1_')
            os.makedirs(root_dir, exist_ok=True)
            
            for file_path in glob.glob(os.path.join(subfolder_path, '*')):
                if os.path.isfile(file_path):
                    try:
                        shutil.move(file_path, os.path.join(root_dir, os.path.basename(file_path)))
                    except FileNotFoundError as e:
                        print(f"文件未找到: {e.filename}")

    def write_cluster_results_to_md(md_filepath: str, subfolder_path: str, image_info_list: List[Dict[str, Optional[str]]]):
        with open(md_filepath, 'a', encoding='utf-8') as md_file:
            md_file.write(f"# 聚類結果 - {subfolder_path}\n")
            md_file.write(f"總圖片數: {len(image_info_list)}\n")
            clusters = {}
            clusters_prompts = []
            cluster_prefixes = ['costume_', 'appearance_', 'scene_']
            for cluster_prefix in cluster_prefixes:
                for info in image_info_list:
                    cluster_name_key = f'{cluster_prefix}cluster_name'
                    cluster_prompt_key = f'{cluster_prefix}cluster_prompt'
                    if info[cluster_name_key] is not None:
                        if info[cluster_name_key] not in clusters:
                            clusters[info[cluster_name_key]] = {
                                'count': 0,
                                'prompt': info[cluster_prompt_key],
                                'files': []
                            }
                        clusters[info[cluster_name_key]]['count'] += 1
                        

            # 排序非 None 的聚類名
            sorted_cluster_names = natsorted(clusters.keys())
            print("\n")
            for cluster_name in sorted_cluster_names:
                cluster_data = clusters[cluster_name]
                cluster_data['prompt']
                print(f"最終聚類名稱: {cluster_name}")
                print(f"聚類標籤: {cluster_data['prompt']}")
                print(f"聚類張數: {cluster_data['count']}")
                print("")    
                md_file.write(f"## {cluster_name}\n")
                md_file.write(f"{cluster_data['prompt']}\n")
                md_file.write(f"聚類張數: {cluster_data['count']}\n")
                md_file.write("\n")
                clusters_prompts.append(cluster_data['prompt'])
            
            for cluster_prefix in cluster_prefixes:
                md_file.write(f"{cluster_prefix}dymatic_prompt : {{{'|'.join(clusters_prompts)}}}  \n")     
            
    subfolder_name = os.path.basename(subfolder_path)
    if "_" not in subfolder_name or not subfolder_name.split("_")[0].isdigit() or ' extra ' in subfolder_name:
        print(f"跳過不符合命名規則的子文件夾: {subfolder_path}")
        return

    repeats = int(subfolder_name.split("_")[0])
    name_from_folder = subfolder_name.split('_')[1].replace('_', ' ').strip()

    print(f"處理子文件夾: {subfolder_name}")
    image_info_list = read_images_and_tags(subfolder_path)
    if not image_info_list or len(image_info_list) < 3:
        print(f"子文件夾 {subfolder_path} 沒有有效的標籤，跳過該文件夾。")
        return

    solo_info_list = [info for info in image_info_list if 'solo' in info['all_tags'] and 'completely nude' not in info['all_tags']]
    costume_tags_list = [info['costume'] for info in solo_info_list]
    n_clusters = min(300, math.ceil(len(solo_info_list) / 5) + 1)

    print("開始聚類...")
    if len(costume_tags_list) > 0:
        process_clustering(solo_info_list, costume_tags_list, n_clusters, 'costume_', args)
        print("服裝聚類完成")

    if args.cluster_appearance: 
        appearance_tags_list = [info['appearance'] for info in solo_info_list]
        if len(appearance_tags_list) > 0:
            n_clusters = min(300, math.ceil(len(solo_info_list) / 5) + 1)
            process_clustering(solo_info_list, appearance_tags_list, n_clusters, 'appearance_', args)
            print("外型聚類完成")
        
    if args.cluster_scene:
        scene_tags_list = [info['scene'] for info in image_info_list]
        if len(scene_tags_list) > 0:
            n_clusters = min(300, math.ceil(len(image_info_list) / 10) + 1)        
            process_clustering(image_info_list, scene_tags_list, n_clusters, 'scene_', args)
            print("場景聚類完成")
        
    if not args.dry_run:
        for info in image_info_list:
            insert_cluster_text_to_txt(info, args)

    copy_or_move_clusters(subfolder_path, subfolder_name, image_info_list, repeats, name_from_folder, args)

    write_cluster_results_to_md(md_filepath, subfolder_path, image_info_list)

def main():
    parser = argparse.ArgumentParser(description="聚類分析腳本")
    parser.add_argument('--cluster_model_name', choices=['K-Means聚類', 'Spectral譜聚類', 'Agglomerative層次聚類', 'OPTICS聚類'], default='Agglomerative層次聚類', help='聚類模型名稱')
    parser.add_argument('--dry_run', action='store_true', help='不輸出文本')
    parser.add_argument('--move_cluster', action='store_true', help='移動到子資料夾的聚類文件夾')
    parser.add_argument('--copy_cluster', action='store_true', help='複製到子資料夾的extra文件夾')
    parser.add_argument('--cluster_appearance', action='store_true', help='對appearance標籤聚類')
    parser.add_argument('--cluster_scene', action='store_true', help='對scenes標籤聚纇')
    parser.add_argument('--naming_mode', choices=['manual', 'auto', 'gpt4o'], default='auto', help='命名模式：手動檢查或自動命名')
    parser.add_argument('--dir_mode', choices=['costume', 'appearance', 'scene'], default='costume', help='檔案模式：依照服裝、外表或場景聚類服裝')
    args = parser.parse_args()
    
    if args.naming_mode == "gpt4o" and (not GPT4O_API_KEY or GPT4O_API_KEY == "YOUR_GPT4O_API_KEY"):
        raise ValueError("API Key is not set. Please set the OPENAI_API_KEY environment variable.")

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_zip_path = os.path.join(parent_dir, f"backup_{timestamp}.zip")

        with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            for root, _, files in os.walk(parent_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, parent_dir)
                        backup_zip.write(file_path, arcname)
        print(f"文本備份完成: {backup_zip_path}")

    md_filepath = os.path.join(parent_dir, "cluster_results.md")
    with open(md_filepath, 'w', encoding='utf-8') as md_file:
        md_file.write("# 聚類結果\n\n")      

    for subfolder in subfolders:
        try:
            process_subfolder(subfolder, args, md_filepath)
        except:
            print(f"{subfolders}處理出錯 略過")

if __name__ == "__main__":
    main()
