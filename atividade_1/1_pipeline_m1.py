import requests
import json
import pandas as pd
from llama_cpp import Llama
from bert_score import score
import difflib
import gc
import warnings
warnings.filterwarnings('ignore')

print("1. Baixando o Dataset M1...")
url_m1 = "https://raw.githubusercontent.com/Itaymanes/K-QA/main/dataset/questions_w_answers.jsonl"
todas_questoes = [json.loads(line) for line in requests.get(url_m1).text.strip().split('\n')]
suas_questoes = todas_questoes[168:185] # Questões 169 a 185

resultados = [{'ID_Questao': i + 169, 'Pergunta': q.get('Question', ''), 'Gabarito': q.get('Free_form_answer', '')} for i, q in enumerate(suas_questoes)]
modelos_gguf = [("Llama-3", "./modelos/llama3.gguf"), ("Mistral", "./modelos/mistral.gguf"), ("Phi-3", "./modelos/phi3.gguf")]

print("\n2. Iniciando Inferência na RTX 3060...")
for nome_modelo, caminho in modelos_gguf:
    print(f"Carregando {nome_modelo}...")
    llm = Llama(model_path=caminho, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    
    for i, item in enumerate(suas_questoes):
        try:
            resposta = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert medical professional. Answer the medical question clearly, technically, and based on evidence."},
                    {"role": "user", "content": item.get('Question', '')}
                ], max_tokens=400, temperature=0.2
            )
            resultados[i][f'Resposta_{nome_modelo}'] = resposta["choices"][0]["message"]["content"]
        except Exception as e:
            resultados[i][f'Resposta_{nome_modelo}'] = f"Erro: {e}"
    del llm
    gc.collect()

df = pd.DataFrame(resultados)
gabaritos = df['Gabarito'].tolist()

print("\n3. Calculando BERTScore e F1-Token...")
def calc_f1(g, r):
    g_tok, r_tok = set(str(g).lower().split()), set(str(r).lower().split())
    if not g_tok or not r_tok: return 0.0
    intsec = g_tok.intersection(r_tok)
    p, rec = len(intsec) / len(g_tok), len(intsec) / len(r_tok)
    return 0.0 if p + rec == 0 else 2 * (p * rec) / (p + rec)

for nome, _ in modelos_gguf:
    resp = df[f'Resposta_{nome}'].tolist()
    P, R, F1_bert = score(resp, gabaritos, lang="en", verbose=False)
    df[f'BERTScore_{nome}'] = F1_bert.numpy()
    df[f'TokenF1_{nome}'] = [calc_f1(g, r) for g, r in zip(resp, gabaritos)]

print("\n4. Calculando Desvio entre os Modelos...")
def calc_desvio(t1, t2):
    sim = difflib.SequenceMatcher(None, str(t1).lower(), str(t2).lower()).ratio()
    return round((1 - sim) * 100, 2)

for idx, row in df.iterrows():
    df.at[idx, 'Desvio_Llama_Mistral(%)'] = calc_desvio(row.get('Resposta_Llama-3'), row.get('Resposta_Mistral'))
    df.at[idx, 'Desvio_Llama_Phi3(%)'] = calc_desvio(row.get('Resposta_Llama-3'), row.get('Resposta_Phi-3'))
    df.at[idx, 'Desvio_Mistral_Phi3(%)'] = calc_desvio(row.get('Resposta_Mistral'), row.get('Resposta_Phi-3'))

df.to_excel('M1_Fase1_Sergio.xlsx', index=False)
print("Concluído! Salvo em 'M1_Fase1_Sergio.xlsx'.")