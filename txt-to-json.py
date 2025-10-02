import json
import re
from bs4 import BeautifulSoup

def html_to_json(html_file_path, json_output_path):
    """
    Converte o HTML das questões para JSON com a estrutura especificada.
    """
    # Lê o arquivo HTML
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse do HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Encontra todos os articles (cada um é uma questão)
    articles = soup.find_all('article')
    
    questoes = []
    
    for article in articles:
        # Inicializa a estrutura da questão
        questao = {
            "enunciado_txt": "",
            "alternativas": {
                "alternativa_a_txt": "",
                "alternativa_b_txt": "",
                "alternativa_c_txt": "",
                "alternativa_d_txt": "",
                "alternativa_e_txt": ""
            },
            "alternativa_correta": ""
        }
        
        enunciado_parts = []
        
        # Processa todos os elementos dentro do article
        for element in article.children:
            if element.name == 'p':
                # Verifica se é uma alternativa
                class_attr = element.get('class', [])
                
                # Verifica se tem classe de alternativa
                is_alternative = False
                for cls in class_attr:
                    if cls.startswith('alternativa_'):
                        # É uma alternativa específica
                        alternativa_key = f"{cls}_txt"
                        # Pega o HTML interno do <p>
                        questao["alternativas"][alternativa_key] = str(element)
                        is_alternative = True
                        break
                
                # Se não é alternativa, faz parte do enunciado
                if not is_alternative:
                    enunciado_parts.append(str(element))
            else:
                # Outros elementos (como quebras de linha) também vão pro enunciado
                if str(element).strip():
                    enunciado_parts.append(str(element))
        
        # Junta todas as partes do enunciado
        questao["enunciado_txt"] = "".join(enunciado_parts)
        
        # Só adiciona a questão se tiver enunciado
        if questao["enunciado_txt"].strip():
            questoes.append(questao)
    
    # Cria a estrutura final
    resultado = {
        "questoes": questoes
    }
    
    # Salva em JSON
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    print(f"JSON salvo com sucesso em: {json_output_path}")
    print(f"Total de questões processadas: {len(questoes)}")
    
    return resultado


# Exemplo de uso
if __name__ == "__main__":
    html_file = "prova_enem.txt"  # Arquivo HTML de entrada
    json_file = "questoes_enem.json"  # Arquivo JSON de saída
    
    resultado = html_to_json(html_file, json_file)
    
    # Exibe uma prévia da primeira questão (se existir)
    if resultado["questoes"]:
        print("\n--- Prévia da primeira questão ---")
        primeira = resultado["questoes"][0]
        print(f"Enunciado: {primeira['enunciado_txt'][:200]}...")
        print(f"Alternativa A: {primeira['alternativas']['alternativa_a_txt'][:100]}...")