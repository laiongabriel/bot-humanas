import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import base64
import re
import json
from collections import Counter
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# Lista de trechos indesejados (considerando variações comuns)
# -------------------------------------------------------------------
textos_indesejados = [
    "LINGUAGENS, CÓDIGOS E SUAS TECNOLOGIAS Questões de 01 a 45",
    "CIÊNCIAS HUMANAS E SUAS TECNOLOGIAS Questões de 46 a 90",
    "CIÊNCIAS DA NATUREZA E SUAS TECNOLOGIAS Questões de 91 a 135",
    "MATEMÁTICA E SUAS TECNOLOGIAS Questões de 136 a 180",
    "Questões de 01 a 05 (opção espanhol)",
    "19 CIÊNCIAS HUMANAS E SUAS TECNOLOGIAS",
    "Questões de 46 a 90",
    "CIÊNCIAS HUMANAS E SUAS TECNOLOGIAS",
    "• LINGUAGENS, CÓDIGOS E SUAS TECNOLOGIAS E REDAÇÃO • 1",
    "DIA • CADERNO 1 • AZUL 5",
    "LINGUAGENS, CÓDIGOS E SUAS TECNOLOGIAS E REDAÇÃO",
    "LINGUAGENS, CÓDIGOS E SUAS TECNOLOGIAS E REDAÇÃO • 1",
    "DIA • CADERNO 1 • AZUL •",
    "DIA • CADERNO 1 • AZUL ",
    "DIA • CADERNO [1-3][0-9]",
    "DIA • CADERNO [0-9]",
    "LC - 1º dia | Caderno 1 - AZUL - Página 3",
    "LC - 1º dia | Caderno 1 - AZUL - Página [0-9]",
    "LC - 1º dia | Caderno 1 - AZUL - Página [0-9][0-9]",
    "CIÊNCIAS DA NATUREZA E SUAS TECNOLOGIAS",
    "Questões de 91 a 135",
    "MATEMÁTICA E SUAS TECNOLOGIAS",
    "Questões de 136 a 180",
    "º",
    "DIA •",
    "[0-9][0-9] – –",
    "[0-9] – –",
    "LC - 1º"
]

def trecho_indesejado(texto):
    txt = texto.strip()
    for undes in textos_indesejados:
        if txt.startswith(undes) or undes in txt:
            return True
    return False

# -------------------------------------------------------------------
# Função auxiliar: calcula a razão de sobreposição entre dois bboxes.
# -------------------------------------------------------------------
def get_overlap_ratio(bbox_text, bbox_img):
    x0 = max(bbox_text[0], bbox_img[0])
    y0 = max(bbox_text[1], bbox_img[1])
    x1 = min(bbox_text[2], bbox_img[2])
    y1 = min(bbox_text[3], bbox_img[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter_area = (x1 - x0) * (y1 - y0)
    text_area = (bbox_text[2] - bbox_text[0]) * (bbox_text[3] - bbox_text[1])
    return inter_area / text_area if text_area > 0 else 0

# -------------------------------------------------------------------
# Função auxiliar: detecta se existe uma linha vertical na imagem
# -------------------------------------------------------------------
def detect_vertical_line(image_path, min_line_length_ratio=0.8, angle_threshold=10, margin_ratio=0.1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=int(image.shape[0] * min_line_length_ratio), maxLineGap=10)
    if lines is None:
        print("Nenhuma linha detectada.", "pagina: " + image_path)
        return None
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy == 0:
                continue
            angle = np.degrees(np.arctan(dx / dy))
            if angle < angle_threshold:
                if x1 > image.shape[1] * margin_ratio and x1 < image.shape[1] * (1 - margin_ratio):
                    vertical_lines.append((x1, y1, x2, y2))
    if not vertical_lines:
        return None
    xs = [(line[0] + line[2]) / 2 for line in vertical_lines]
    vertical_line_x = int(np.median(xs))
    return vertical_line_x

def is_completely_inside(inner_bbox, outer_bbox):
    return (inner_bbox[0] >= outer_bbox[0] and
            inner_bbox[1] >= outer_bbox[1] and
            inner_bbox[2] <= outer_bbox[2] and
            inner_bbox[3] <= outer_bbox[3])

# -------------------------------------------------------------------
# Função auxiliar: converte uma imagem para base64
# -------------------------------------------------------------------
def image_to_base64_data(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# -------------------------------------------------------------------
# Função auxiliar: formata um texto de acordo com a fonte
# -------------------------------------------------------------------
def format_text(span):
    text = span.get("text", "").strip()
    if not text:
        return ""
    fonte = span.get("font", "")
    if fonte == "Arial-ItalicMT":
        return f"<i>{text}</i>"
    elif fonte == "Arial-BoldMT":
        return f"<strong>{text}</strong>"
    else:
        return text

# -------------------------------------------------------------------
# 1. Extrai as páginas do PDF como imagens
# -------------------------------------------------------------------
def extract_pages_as_images(pdf_path, output_dir="output_images", scale_factor=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    doc = fitz.open(pdf_path)
    image_paths = []
    mat = fitz.Matrix(scale_factor, scale_factor)
    for page_number in range(1, len(doc)-1):
        page_text = doc[page_number].get_text()
        if page_text.startswith("INSTRUÇÕES PARA A REDAÇÃO"):
            continue
        page = doc[page_number]
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths

# -------------------------------------------------------------------
# 2. Processa e recorta as imagens que possuem borda roxa
# -------------------------------------------------------------------
def process_and_cut_images(input_images, output_dir="cut-imgs", scale_factor=3):
    import os, cv2, numpy as np
    cropped_images = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([170, 255, 255])

    for input_image_path in input_images:
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        cropped_images[base_name] = []
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Erro ao carregar a imagem: {input_image_path}")
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        purple_region = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(purple_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median_val = np.median(blurred)
        lower_threshold = int(max(0, 0.7 * median_val))
        upper_threshold = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        image_counter = 0
        min_width = image.shape[1] * 0.02
        min_height = image.shape[0] * 0.02
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width and h > min_height:
                cropped_image = image[y:y + h, x:x + w]
                cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                purple_mask = cv2.inRange(cropped_hsv, lower_purple, upper_purple)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
                cropped_image[purple_mask > 0] = [255, 255, 255]
                output_path = os.path.join(output_dir, f'{base_name}_cut_{image_counter}.png')
                cv2.imwrite(output_path, cropped_image)
                pdf_bbox = (x / scale_factor, y / scale_factor, (x + w) / scale_factor, (y + h) / scale_factor)
                raw_coords = (x, y, w, h)
                cropped_images[base_name].append((output_path, pdf_bbox, raw_coords))
                image_counter += 1
    print(f"Imagens recortadas salvas em '{output_dir}'.")
    return cropped_images

# -------------------------------------------------------------------
# 3. Extrai o texto do PDF e insere tags <img> para os blocos de imagem
# -------------------------------------------------------------------
def pdf_to_txt_with_img_tags(pdf_path, txt_path, cropped_images_mapping, overlap_threshold=0.3, scale_factor=3):
    import os, re
    from collections import Counter
    import fitz

    doc = fitz.open(pdf_path)

    most_common_size = None
    sample_page_index = 3
    if len(doc) > sample_page_index:
        sample_page = doc.load_page(sample_page_index)
        sample_blocks = sample_page.get_text("dict")["blocks"]
        sizes = []
        for block in sample_blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        texto = re.sub(r'[\t\u0007]+', ' ', span.get("text", ""))
                        tamanho = span.get("size", 0)
                        fonte = span.get("font", "")
                        if ((fonte == "Arial-BoldMT" and tamanho == 8) or
                                tamanho < 5 or tamanho > 11 or
                                trecho_indesejado(texto)):
                            continue
                        sizes.append(tamanho)
        if sizes:
            most_common_size = Counter(sizes).most_common(1)[0][0]

    output_lines = []

    for page_num in range(1, len(doc) - 1):
        page = doc.load_page(page_num)
        base_name = f"page_{page_num + 1}"

        page_text = page.get_text()
        if "INSTRUÇÕES PARA A REDAÇÃO" in page_text:
            continue

        page_image_path = os.path.join("output_images", f"page_{page_num + 1}.png")
        vertical_line = detect_vertical_line(page_image_path)
        vertical_line_pdf = vertical_line / scale_factor if vertical_line is not None else None

        blocks = page.get_text("dict")["blocks"]
        items = []

        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    line_text = ""
                    alternativa_found = False
                    column = 0
                    if line["spans"]:
                        bbox0 = line["spans"][0].get("bbox", [0, 0, 0, 0])
                        x0 = bbox0[0]
                        if vertical_line_pdf is not None and x0 > vertical_line_pdf:
                            column = 1
                    skip_line = False
                    for span in line["spans"]:
                        span_bbox = span.get("bbox", None)
                        if span_bbox and base_name in cropped_images_mapping:
                            for (_, purple_bbox, *rest) in cropped_images_mapping[base_name]:
                                if get_overlap_ratio(span_bbox, purple_bbox) > overlap_threshold:
                                    skip_line = True
                                    break
                        if skip_line:
                            break
                        texto = re.sub(r'[\t\u0007]+', ' ', span.get("text", ""))
                        tamanho = span.get("size", 0)
                        fonte = span.get("font", "")
                        if fonte == "BundesbahnPiStd-1":
                            alternativa_found = True
                        if (((fonte == "Arial-BoldMT" and tamanho == 8) or
                             tamanho < 5 or tamanho > 11 or
                             trecho_indesejado(texto) or
                             re.match(r'^[0-9][0-9] – –$', texto) or
                             re.match(r'^[0-9] – –$', texto)) or
                                (fonte == "ArialMT" and tamanho == 8) or
                                (fonte == "ArialMT" and tamanho == 7) or
                                (fonte == "Arial-BoldMT" and tamanho == 8)):
                            continue
                        if texto.strip():
                            formatted = format_text(span)
                            if most_common_size is not None and tamanho <= (most_common_size - 1):
                                formatted = f"<small>{formatted}</small>"
                            line_text += formatted + " "
                    if not skip_line and line_text.strip():
                        first_span_bbox = line["spans"][0].get("bbox", [0, 0, 0, 0])
                        y_val = first_span_bbox[1]
                        x_val = first_span_bbox[0]
                        if alternativa_found:
                            content = f'<p class="alternativa">{line_text.strip()}</p>'
                        else:
                            content = f'<p>{line_text.strip()}</p>'
                        items.append({
                            "y": y_val,
                            "x": x_val,
                            "column": column,
                            "type": "text",
                            "content": content
                        })

        if base_name in cropped_images_mapping:
            for entry in cropped_images_mapping[base_name]:
                if len(entry) == 3:
                    img_path, old_bbox, raw_coords = entry
                    x, y, w, h = raw_coords
                    bbox = (x / scale_factor, y / scale_factor, (x + w) / scale_factor, (y + h) / scale_factor)
                else:
                    img_path, bbox = entry
                center_x = (bbox[0] + bbox[2]) / 2
                column = 0
                if vertical_line_pdf is not None and center_x > vertical_line_pdf:
                    column = 1
                y_val = bbox[1]
                items.append({
                    "y": y_val,
                    "x": bbox[0],
                    "column": column,
                    "type": "img",
                    "content": (img_path, bbox)
                })

        if vertical_line_pdf is not None:
            items.sort(key=lambda i: (i["column"], i["y"], i["x"]))
        else:
            items.sort(key=lambda i: (i["y"], i["x"]))

        merged_items = []
        if items:
            current_item = items[0]
            for item in items[1:]:
                if (current_item["type"] == "text" and item["type"] == "text" and
                    current_item["column"] == item["column"] and
                    abs(item["y"] - current_item["y"]) < 5):
                    current_content = current_item["content"].replace("<p>", "").replace("</p>", "")
                    item_content = item["content"].replace("<p>", "").replace("</p>", "")
                    current_item["content"] = f"<p>{current_content} {item_content}</p>"
                else:
                    merged_items.append(current_item)
                    current_item = item
            merged_items.append(current_item)
        else:
            merged_items = items

        page_output = []
        for item in merged_items:
            if item["type"] == "text":
                page_output.append(item["content"])
            elif item["type"] == "img":
                img_path, bbox = item["content"]
                img_data_uri = image_to_base64_data(img_path)
                pdf_width = bbox[2] - bbox[0]
                pdf_height = bbox[3] - bbox[1]
                page_output.append(
                    f'<p><img src="{img_data_uri}" alt="Elemento com borda roxa" '
                    f'style="width:{pdf_width}pt; height:{pdf_height}pt; display:block;" /></p>'
                )
        output_lines.extend(page_output)
        output_lines.append("\n")

    html_output = "<html><head><meta charset='utf-8'></head><body>" + "\n".join(output_lines) + "</body></html>"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    print("HTML com tags de imagem e formatação salvo em:", txt_path)

# -------------------------------------------------------------------
# 4. Converte o HTML para JSON estruturado
# -------------------------------------------------------------------
def html_to_questoes_json(html_path, json_output_path):
    """
    Converte o HTML gerado com questões do ENEM para o formato JSON estruturado.
    
    Args:
        html_path: Caminho do arquivo HTML de entrada
        json_output_path: Caminho do arquivo JSON de saída
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    paragrafos = soup.find_all('p')
    
    print(f"Total de parágrafos encontrados: {len(paragrafos)}")
    
    # Debug: mostra os primeiros 5 parágrafos
    print("\nPrimeiros 5 parágrafos:")
    for i, p in enumerate(paragrafos[:5]):
        print(f"{i}: {str(p)[:200]}")
    
    questoes = []
    questao_atual = None
    enunciado_parts = []
    alternativas_encontradas = {}
    
    questao_pattern = re.compile(r'^<strong>\s*(\d+)\s*</strong>')
    alternativa_pattern = re.compile(r'^\s*([A-E])\s+', re.IGNORECASE)
    
    for p in paragrafos:
        p_html = str(p)
        p_text = p.get_text().strip()
        
        match_questao = questao_pattern.search(p_html)
        
        if match_questao:
            if questao_atual is not None:
                questao_atual['enunciado_txt'] = ''.join(enunciado_parts).strip()
                questao_atual['alternativas'] = alternativas_encontradas.copy()
                questoes.append(questao_atual)
            
            numero_questao = match_questao.group(1)
            questao_atual = {
                'numero': int(numero_questao),
                'enunciado_txt': '',
                'alternativas': {},
                'alternativa_correta': ''
            }
            enunciado_parts = []
            alternativas_encontradas = {}
            
            p_html_sem_numero = questao_pattern.sub('', p_html)
            if p_html_sem_numero.strip() not in ['<p></p>', '<p> </p>']:
                enunciado_parts.append(p_html_sem_numero)
        
        elif 'class="alternativa"' in p_html and questao_atual is not None:
            conteudo_alternativa = p_html.replace('<p class="alternativa">', '').replace('</p>', '').strip()
            
            match_letra = alternativa_pattern.match(p_text)
            
            if match_letra:
                letra = match_letra.group(1).upper()
                conteudo_sem_letra = alternativa_pattern.sub('', conteudo_alternativa).strip()
                
                chave = f'alternativa_{letra.lower()}_txt'
                alternativas_encontradas[chave] = conteudo_sem_letra
        
        elif questao_atual is not None and not alternativas_encontradas:
            if p_html.strip() not in ['<p></p>', '<p> </p>']:
                enunciado_parts.append(p_html)
    
    if questao_atual is not None:
        questao_atual['enunciado_txt'] = ''.join(enunciado_parts).strip()
        questao_atual['alternativas'] = alternativas_encontradas.copy()
        questoes.append(questao_atual)
    
    for q in questoes:
        if 'numero' in q:
            del q['numero']
    
    resultado = {
        'questoes': questoes
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    print(f"JSON gerado com {len(questoes)} questões em: {json_output_path}")
    return resultado

# -------------------------------------------------------------------
# Execução principal
# -------------------------------------------------------------------
if __name__ == "__main__":
    pdf_path = "provas/dia1_caderno1_azul.pdf"
    txt_path = "prova_enem.txt"
    json_path = "questoes_enem.json"
    scale_factor = 3

    # 1. Extrai as páginas do PDF como imagens
    input_images = extract_pages_as_images(pdf_path, output_dir="output_images", scale_factor=scale_factor)
    
    # 2. Processa e recorta as imagens com borda roxa
    cropped_images_mapping = process_and_cut_images(input_images, output_dir="cut-imgs", scale_factor=scale_factor)
    
    # 3. Extrai o texto do PDF e gera HTML
    pdf_to_txt_with_img_tags(pdf_path, txt_path, cropped_images_mapping, overlap_threshold=0.5, scale_factor=scale_factor)
    
    # 4. Converte o HTML para JSON estruturado
    html_to_questoes_json(txt_path, json_path)