import fitz
import pandas as pd
import re

print("Abrindo o usmle_report.pdf...")
doc = fitz.open("usmle_report.pdf")
texto_completo = ""

for pagina in doc:
    texto_completo += pagina.get_text()

print("Mapeando a estrutura do documento ignorando a numeração interna...")

# Procura exatamente o padrão "Question [numero].[numero] ("
padrao_cabecalho = r'Question \d+\.\d+ \('
matches = list(re.finditer(padrao_cabecalho, texto_completo))

todas_questoes = []

# Varre cada bloco encontrado
for i in range(len(matches)):
    inicio = matches[i].start()
    # O fim deste bloco é o início do próximo. Se for a última questão, vai até o fim do texto.
    fim = matches[i+1].start() if i + 1 < len(matches) else len(texto_completo)
    
    bloco = texto_completo[inicio:fim]
    
    # Extrai o gabarito oficial (A, B, C, D ou E)
    match_gabarito = re.search(r'Correct Response:\s*([A-E])', bloco)
    
    if match_gabarito:
        gabarito = match_gabarito.group(1).strip()
        # A pergunta é todo o texto do bloco até a palavra "Correct Response:"
        pergunta = bloco.split('Correct Response:')[0].strip()
        
        todas_questoes.append({
            "Question": pergunta,
            "Answer": gabarito
        })

print(f"Total de questões sequenciais encontradas no PDF: {len(todas_questoes)}")

# Isola exatamente as suas questões pela posição real no documento!
# Questões 271 a 297 equivalem aos índices 270 a 296 no Python.
suas_questoes = todas_questoes[270:297]
print(f"Extraindo o seu lote... Total isolado: {len(suas_questoes)}")

df = pd.DataFrame(suas_questoes)
df.to_excel("dataset_m2.xlsx", index=False)
print("\nPronto! O arquivo 'dataset_m2.xlsx' foi gerado perfeitamente limpo e estruturado.")