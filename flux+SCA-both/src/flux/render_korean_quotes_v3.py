import csv
import os
import random
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import math

# ===========================
# Color & contrast utilities
# ===========================

def srgb_to_linear(c: float) -> float:
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    R = srgb_to_linear(r)
    G = srgb_to_linear(g)
    B = srgb_to_linear(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

def contrast_ratio(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    L1 = relative_luminance(c1)
    L2 = relative_luminance(c2)
    L1, L2 = (L1, L2) if L1 >= L2 else (L2, L1)
    return (L1 + 0.05) / (L2 + 0.05)

def pick_contrasting_text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    white = (255, 255, 255)
    black = (0, 0, 0)
    cw = contrast_ratio(bg, white)
    cb = contrast_ratio(bg, black)
    return white if cw >= cb else black

def random_bg_color(min_dist_to_gray: int = 20) -> Tuple[int, int, int]:
    while True:
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if abs(rgb[0]-128) + abs(rgb[1]-128) + abs(rgb[2]-128) > min_dist_to_gray:
            return rgb

def rgb_to_name(rgb: Tuple[int,int,int]) -> str:
    """Rudimentary color naming for prompts."""
    r,g,b = rgb
    # Grayscale checks
    if max(r,g,b) < 35: return "black"
    if min(r,g,b) > 220: return "white"
    # Convert to HSV
    mx = max(r,g,b); mn = min(r,g,b); diff = mx - mn
    if mx == 0: sat = 0
    else: sat = diff / mx
    if diff == 0:
        return "gray"
    if mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    v = mx / 255.0

    if v < 0.25: return "dark " + ("gray" if sat < 0.2 else "color")
    if sat < 0.2: return "gray"

    if   0 <= h < 15:   name = "red"
    elif 15 <= h < 45:  name = "orange"
    elif 45 <= h < 70:  name = "yellow"
    elif 70 <= h < 170: name = "green"
    elif 170 <= h < 200: name = "cyan"
    elif 200 <= h < 255: name = "blue"
    elif 255 <= h < 290: name = "purple"
    elif 290 <= h < 330: name = "magenta"
    else: name = "red"
    return name

# ===========================
# Text & HTML-like tagging
# ===========================

class Token:
    def __init__(self, text: str, styles: Dict[str,bool]):
        self.text = text
        self.styles = styles  # {'b':bool, 'i':bool, 'u':bool}

def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()

def text_w_h(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

# ===========================
# Style Generation Logic (수정됨)
# ===========================

# FonTS 논문 스타일의 ETC-Tokens 정의
ETC_TOKENS_MAP = {
    'b': (r'<\bold>', r'</bold>'),
    'i': (r'<\italic>', r'</italic>'),
    'u': (r'<\underline>', r'</underline>'),
}

def generate_styled_variations(
    original_text: str, 
    num_positions: int = 5, 
    styles_to_apply: Tuple[str, ...] = ("b", "i", "u")
) -> List[Dict[str, any]]:
    """
    원본 텍스트와 (N개 위치 * M개 스타일) 변형을 생성합니다.
    """
    words = original_text.split(" ")
    variations = []

    # --- 1. 원본 (스타일 없음) 버전 추가 ---
    original_tokens = []
    for w in words:
        original_tokens.append(Token(w, {'b': False, 'i': False, 'u': False}))
    
    variations.append({
        'prompt_text': original_text,  # ETC-Token 없는 순수 텍스트
        'html_text': original_text,    # 렌더링을 위한 HTML-like 텍스트
        'tokens': original_tokens,
        'variation_name': 'original'
    })

    # --- 2. 스타일 적용할 위치 선정 ---
    valid_indices = [i for i, w in enumerate(words) if w.strip()]
    
    if len(valid_indices) < num_positions:
        chosen_indices = valid_indices
    else:
        chosen_indices = random.sample(valid_indices, num_positions)

    # --- 3. (5개 위치 * 3개 스타일 = 15개) 변형 생성 ---
    for style_tag in styles_to_apply:  # "b", "i", "u"
        start_token, end_token = ETC_TOKENS_MAP[style_tag] # ETC-Token 가져오기
        
        for word_idx in chosen_indices:
            
            html_parts = []
            prompt_parts = []
            token_list = []
            
            for i, w in enumerate(words):
                styles = {'b': False, 'i': False, 'u': False}
                
                # 렌더링을 위한 HTML 태그 (이미지 렌더링 엔진이 인식)
                html_part = w
                
                # 프롬프트를 위한 ETC-Token
                prompt_part = w
                
                if i == word_idx:
                    styles[style_tag] = True
                    # 렌더링 엔진을 위해 HTML 태그 사용
                    html_part = f"<{style_tag}>{w}</{style_tag}>"
                    # 모델 훈련 프롬프트를 위해 ETC-Token 사용
                    prompt_part = f"{start_token} {w} {end_token}" 
                
                html_parts.append(html_part)
                prompt_parts.append(prompt_part)
                token_list.append(Token(w, styles))
            
            html_text = " ".join(html_parts)
            prompt_text = " ".join(prompt_parts) # ETC-Token이 포함된 프롬프트 텍스트
            variation_name = f"pos{word_idx}_{style_tag}"
            
            variations.append({
                'prompt_text': prompt_text,
                'html_text': html_text,
                'tokens': token_list,
                'variation_name': variation_name
            })
            
    return variations

# ===========================
# Core Rendering Engine
# ===========================

def compute_line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent

def wrap_tokens_to_width(draw: ImageDraw.ImageDraw, tokens: List[Token], font: ImageFont.FreeTypeFont, max_width: int) -> List[List[Token]]:
    space_w, _ = text_w_h(draw, " ", font)
    lines: List[List[Token]] = []
    current: List[Token] = []
    line_w = 0
    for t in tokens:
        tw, _ = text_w_h(draw, t.text, font)
        add_w = tw if not current else (space_w + tw)
        if current and (line_w + add_w) > max_width:
            lines.append(current)
            current = [t]
            line_w = tw
        else:
            current.append(t)
            line_w += add_w if current[:-1] else tw
    if current:
        lines.append(current)
    return lines

def draw_token(draw: ImageDraw.ImageDraw, x: int, y: int, token: Token, font: ImageFont.FreeTypeFont, fill: Tuple[int,int,int]):
    text = token.text
    stroke_w = 1 if token.styles.get('b') else 0
    if token.styles.get('i'):
        w, h = draw.textbbox((0, 0), text, font=font)[2:4]
        pad = 4
        temp = Image.new("RGBA", (w + 2*pad, h + 2*pad), (0,0,0,0))
        tdraw = ImageDraw.Draw(temp)
        tdraw.text((pad, pad), text, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=fill)
        shear = math.tan(math.radians(12))
        a = 1; b = shear; c = 0
        d = 0; e = 1;     f = 0
        new_w = int(temp.size[0] + abs(b) * temp.size[1])
        sheared = temp.transform((new_w, temp.size[1]), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
        draw._image.paste(sheared, (x, y - pad), sheared)
        if token.styles.get('u'):
            ascent, descent = font.getmetrics()
            uy = y + ascent + 2
            uw = sheared.size[0]
            draw.line((x, uy, x + uw, uy), fill=fill, width=2)
        return
    else:
        draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=fill)
        if token.styles.get('u'):
            ascent, descent = font.getmetrics()
            tw = draw.textbbox((0,0), text, font=font)[2]
            uy = y + ascent + 2
            draw.line((x, uy, x + tw, uy), fill=fill, width=2)

def render_html_tokens(
    img: Image.Image,
    tokens: List[Token],
    align: str,
    padding: int,
    font_path: Optional[str],
    start_font_size: int = 80
) -> None:
    draw = ImageDraw.Draw(img)
    W, H = img.size
    content_w = W - 2 * padding
    content_h = H - 2 * padding
    size = start_font_size
    font = None; lines = None
    while size >= 16:
        font = load_font(font_path, size)
        lines = wrap_tokens_to_width(draw, tokens, font, content_w)
        total_h = len(lines)*compute_line_height(font) + (len(lines)-1)*max(2,int(size*0.2))
        max_w = 0
        for line in lines:
            wsum = 0
            for i, t in enumerate(line):
                tw, _ = text_w_h(draw, t.text, font)
                if i>0:
                    sw, _ = text_w_h(draw, " ", font)
                    wsum += sw
                wsum += tw
            if wsum > max_w:
                max_w = wsum
        if total_h <= content_h and max_w <= content_w:
            break
        size -= 2
    line_spacing = max(2, int(size*0.2))
    total_h = len(lines)*compute_line_height(font) + (len(lines)-1)*line_spacing
    y = padding + (content_h - total_h)//2
    for line in lines:
        wsum = 0
        parts = []
        for i, t in enumerate(line):
            tw, _ = text_w_h(draw, t.text, font)
            sw, _ = text_w_h(draw, " ", font)
            if i>0:
                wsum += sw
            parts.append((t, tw))
            wsum += tw
        if align == 'left':
            x = padding
        elif align == 'center':
            x = padding + (content_w - wsum)//2
        elif align == 'right':
            x = padding + (content_w - wsum)
        else:
            x = padding
        for i, (t, tw) in enumerate(parts):
            if i>0:
                sw, _ = text_w_h(draw, " ", font)
                x += sw
            draw_token(draw, x, y, t, font, img.info.get("text_color", (255,255,255)))
            x += tw
        y += compute_line_height(font) + line_spacing

# ===========================
# ===========================

def render_quotes_from_csv(
    csv_path: str,
    out_dir: str,
    font_path: Optional[str] = None,
    image_size: int = 1024,
    padding_ratio: float = 0.08,
    alignment_mode: str = "random",
    all_alignments: bool = False,
    min_contrast: float = 4.5,
    seed: Optional[int] = None,
    num_style_positions: int = 5,
    rel_image_prefix: str = "./tc-dataset/"
):
    if seed is not None:
        random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    quotes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'quote' not in reader.fieldnames:
            raise ValueError("CSV must have a 'quote' column.")
        for row in reader:
            q = (row.get('quote') or '').strip()
            if q:
                quotes.append(q)

    if not quotes:
        print("No quotes found.")
        return

    # 파일 이름에 사용할 폰트 이름 추출
    if font_path:
        font_name_full = os.path.basename(font_path)
        font_name = os.path.splitext(font_name_full)[0]
    else:
        font_name = "default_font"
        font_name_full = "default_font"

    metadata = []
    total_images = 0

    for idx, text in enumerate(quotes):
        
        variations = generate_styled_variations(
            text, 
            num_positions=num_style_positions
        )
        
        align_list = ['left','center','right'] if all_alignments else [alignment_mode]
        
        for a in align_list:
            align = random.choice(['left','center','right']) if a == 'random' else a

            # 16개의 변형에 대해 루프를 돕니다.
            for var_data in variations:
                
                # 16개 변형마다 새로운 색상을 무작위로 선택
                tries = 0
                while True:
                    bg = random_bg_color()
                    text_color = pick_contrasting_text_color(bg)
                    if contrast_ratio(bg, text_color) >= min_contrast:
                        break
                    tries += 1
                    if tries > 50:
                        text_color = (0,0,0) if contrast_ratio(bg,(0,0,0)) >= contrast_ratio(bg,(255,255,255)) else (255,255,255)
                        break

                html_text = var_data['html_text']
                tokens = var_data['tokens']
                variation_name = var_data['variation_name'] # 예: "original", "pos3_b"

                img = Image.new("RGB", (image_size, image_size), bg)
                img.info["text_color"] = text_color

                padding = int(image_size * padding_ratio)
                render_html_tokens(img, tokens, align, padding, font_path)

                # ETC-Token이 포함된 프롬프트 텍스트
                prompt_text_w_etc = var_data['prompt_text']
                html_text = var_data['html_text'] # 렌더링에 사용되는 HTML 텍스트
                tokens = var_data['tokens']
                variation_name = var_data['variation_name']

                img = Image.new("RGB", (image_size, image_size), bg)
                img.info["text_color"] = text_color

                padding = int(image_size * padding_ratio)
                render_html_tokens(img, tokens, align, padding, font_path)


                # 파일 이름에 폰트 이름을 포함 (예: quote_00001_NanumGothic_center_pos3_b.png)
                base = f"quote_{idx:05d}_{font_name}_{align}_{variation_name}.png"
                
                out_path = os.path.join(out_dir, base)
                img.save(out_path, format="PNG", optimize=True)

                align_eng = {"left":"Left-aligned", "center":"Center-aligned", "right":"Right-aligned"}[align]
                fg_name = rgb_to_name(text_color)
                bg_name = rgb_to_name(bg)

                # 최종 프롬프트 생성 (폰트 정보와 ETC-Token 포함)
                # FonTS의 핵심 프롬프트 구조: [배치/색상/정렬] + [폰트 정보] + [ETC-Token 포함 텍스트]
                prompt = (
                    f'{align_eng} text written in {fg_name} letters on a {bg_name} background, '
                    f'using the font <{font_name}> with the quote "{prompt_text_w_etc}"'
                )

                metadata.append({
                    "prompt": prompt,
                    "image": os.path.join(rel_image_prefix, base).replace("\\","/")
                })
                total_images += 1

    # Write JSONL
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    
    with open(meta_path, "a", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Done. Appended {total_images} images to {out_dir}")
    print(f"Metadata: {meta_path}")

# ===========================
# Main function
# ===========================
def main():
    parser = argparse.ArgumentParser(description="Render Korean quotes with style variations (b/i/u) to images and emit JSONL metadata.")
    parser.add_argument("--csv", required=True, help="Path to CSV file with a 'quote' column (UTF-8).")
    parser.add_argument("--out", required=True, help="Output directory for images & metadata.jsonl.")
    parser.add_argument("--font", default=None, help="Path to TTF/OTF font supporting Korean.")
    parser.add_argument("--size", type=int, default=1024, help="Image size (square). Default 1024.")
    parser.add_argument("--padding_ratio", type=float, default=0.08, help="Padding as ratio of image size. Default 0.08.")
    parser.add_argument("--align", choices=["left","center","right","random"], default="random", help="Text alignment.")
    parser.add_argument("--all_alignments", action="store_true", help="Render left/center/right versions for each quote (multiplies with variations).")
    parser.add_argument("--min_contrast", type=float, default=4.5, help="Minimum contrast ratio for text vs background.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num_positions", type=int, default=5, help="Number of random word positions to apply styles to. Default 5.")
    parser.add_argument("--rel_image_prefix", type=str, default="./tc-dataset/", help="Relative path prefix to store in JSON metadata for 'image'.")
    args = parser.parse_args()

    render_quotes_from_csv(
        csv_path=args.csv,
        out_dir=args.out,
        font_path=args.font,
        image_size=args.size,
        padding_ratio=args.padding_ratio,
        alignment_mode=args.align,
        all_alignments=args.all_alignments,
        min_contrast=args.min_contrast,
        seed=args.seed,
        num_style_positions=args.num_positions,
        rel_image_prefix=args.rel_image_prefix
    )

if __name__ == "__main__":
    main()