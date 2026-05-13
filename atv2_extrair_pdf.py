import fitz
import pandas as pd
import re

# Lista exata dos números das questões que você deseja extrair
numeros_alvo = [
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 
    91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 103, 104, 105, 106
]

print("Abrindo o usmle_report.pdf...")
doc = fitz.open("usmle_report.pdf")
texto_completo = ""

for pagina in doc:
    texto_completo += pagina.get_text()

print("Buscando as questões específicas...")

# Mantém o padrão de busca original
padrao_cabecalho = r'Question \d+\.\d+ \('
matches = list(re.finditer(padrao_cabecalho, texto_completo))

questoes_filtradas = []

for i in range(len(matches)):
    inicio = matches[i].start()
    fim = matches[i+1].start() if i + 1 < len(matches) else len(texto_completo)
    
    bloco = texto_completo[inicio:fim]
    
    match_numero = re.search(r'Question (\d+)\.1', bloco)
    
    if match_numero:
        numero_extraido = int(match_numero.group(1))
        
        if numero_extraido in numeros_alvo:
            # Extrai o gabarito oficial (A, B, C, D ou E)
            match_gabarito = re.search(r'Correct Response:\s*([A-E])', bloco)
            
            if match_gabarito:
                gabarito = match_gabarito.group(1).strip()
                # A pergunta é todo o texto do bloco até a palavra "Correct Response:"
                pergunta = bloco.split('Correct Response:')[0].strip()
                
                questoes_filtradas.append({
                    "Question": pergunta,
                    "Answer": gabarito
                })

print(f"Total de questões encontradas e filtradas: {len(questoes_filtradas)}")

# Gera o Excel apenas com as filtradas
df = pd.DataFrame(questoes_filtradas)
df.to_excel("dataset_m2.xlsx", index=False)

print("\nPronto! O arquivo 'dataset_m2.xlsx' foi gerado apenas com as questões solicitadas.")