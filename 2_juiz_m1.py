import pandas as pd
from llama_cpp import Llama
import re

print("Iniciando LLM-as-a-Judge...")
df = pd.read_excel('M1_Fase1_Sergio.xlsx')
juiz_llm = Llama(model_path="./modelos/llama3.gguf", n_gpu_layers=-1, n_ctx=2048, verbose=False)
modelos = ["Llama-3", "Mistral", "Phi-3"]

for modelo in modelos:
    notas = []
    print(f"Avaliando respostas do {modelo}...")
    for idx, row in df.iterrows():
        prompt = f"Compare the CANDIDATE ANSWER against the GOLDEN STANDARD medical answer. Rate the CANDIDATE ANSWER from 1 to 5 based on medical accuracy.\n1 = Completely wrong.\n3 = Partially correct.\n5 = Perfectly accurate.\nGOLDEN STANDARD: {row['Gabarito']}\nCANDIDATE ANSWER: {row[f'Resposta_{modelo}']}\nOutput ONLY a single integer number from 1 to 5."
        try:
            aval = juiz_llm.create_chat_completion(
                messages=[{"role": "system", "content": "Output only single integer numbers."}, {"role": "user", "content": prompt}],
                max_tokens=5, temperature=0.1
            )
            num = re.search(r'[1-5]', aval["choices"][0]["message"]["content"])
            notas.append(int(num.group()) if num else "Erro")
        except: notas.append("Erro")
    df[f'Nota_Juiz_{modelo}'] = notas

df.to_excel('M1_FINAL_Sergio.xlsx', index=False)
print("Avaliação concluída! Planilha final 'M1_FINAL_Sergio.xlsx' gerada.")