import os
import pandas as pd
import time
import re
from dotenv import load_dotenv
from groq import Groq

# Carregar configuração
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Erro: Chave da Groq não encontrada no .env")
    exit()

client = Groq(api_key=GROQ_API_KEY)

print("1. Lendo os dados das avaliações do CSV...")
try:
    df = pd.read_csv("avaliacoes_consolidadas_equipe5.csv")
    df['nota_do_juiz'] = pd.to_numeric(df['nota_do_juiz'], errors='coerce')
except FileNotFoundError:
    print("Arquivo 'avaliacoes_consolidadas_equipe5.csv' não encontrado.")
    exit()

# Filtrar os Extremos e dividir em lotes de 5
df_ruins = df[df['nota_do_juiz'] == 1].copy()
df_bons = df[df['nota_do_juiz'] == 5].copy()

# Função para quebrar o dataframe em pedaços menores (lotes)
def criar_lotes(dataframe, tamanho_lote=5):
    lotes = []
    for i in range(0, len(dataframe), tamanho_lote):
        lotes.append(dataframe.iloc[i:i+tamanho_lote])
    return lotes

lotes_ruins = criar_lotes(df_ruins)
lotes_bons = criar_lotes(df_bons)

print(f"-> Temos {len(lotes_ruins)} lotes de Erros Críticos e {len(lotes_bons)} lotes de Acertos Perfeitos.")
print("-> O script vai processar 1 lote por minuto para respeitar o limite de Tokens Por Minuto (TPM).\n")

# Cria/Limpa o arquivo antes de começar
with open("relatorio_insights_completo.md", "w", encoding="utf-8") as f:
    f.write("# 🩺 Relatório Extensivo de Auditoria Clínica (Llama-3.3-70B)\n\n")

# Função que envia o lote para a API e lida com pausas
def processar_lote_com_pausa(lote, tipo, numero_lote, total_lotes):
    limite_chars = 250
    contexto = f"=== LOTE DE {tipo.upper()} ===\n"
    
    for _, row in lote.iterrows():
        contexto += f"- Modelo: {row['modelo_avaliado']}\n"
        contexto += f"  Pergunta: {str(row['pergunta'])[:limite_chars]}...\n"
        contexto += f"  Gabarito: {str(row['gabarito'])[:limite_chars]}...\n"
        contexto += f"  Resposta IA: {str(row['resposta_da_ia'])[:limite_chars]}...\n"
        contexto += f"  Juiz: {str(row['justificativa_do_juiz'])[:limite_chars]}...\n\n"

    prompt_sistema = f"""Você é o Diretor Médico Chefe auditando este lote específico de respostas de IA.
Trata-se de um lote focado apenas em {tipo}. 

Faça um resumo analítico (1 ou 2 parágrafos no máximo) sobre:
1. O que há em comum nessas respostas deste lote?
2. Quais são as falhas estruturais (se for nota 1) ou os trunfos clínicos (se for nota 5) específicos destes casos?

Responda em Markdown, de forma concisa e técnica."""

    while True:
        try:
            print(f"Processando Lote {numero_lote}/{total_lotes} ({tipo})...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Aqui está o lote:\n\n{contexto}"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2, 
                max_tokens=600 
            )
            
            analise = chat_completion.choices[0].message.content
            
            # Salva no arquivo abrindo em modo "append" (a)
            with open("relatorio_insights_completo.md", "a", encoding="utf-8") as f:
                f.write(f"## Análise do Lote {numero_lote} ({tipo})\n")
                f.write(f"{analise}\n\n---\n\n")
            
            print(f"[OK] Lote {numero_lote} analisado. Entrando em modo de espera (65s) para limpar TPM...")
            time.sleep(65) # Espera 1 minuto para a Groq zerar o contador
            break # Sai do loop while e vai para o próximo lote

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                # Se for o erro de 100k ara o programa inteiro
                if "100000" in error_msg or "Tokens per day" in error_msg:
                    print("\n[FIM DE EXPEDIENTE] Você atingiu os 100.000 tokens diários!")
                    print("O relatório gerado até agora foi salvo com sucesso.")
                    return "PARAR_TUDO"
                else:
                    # Se for só o erro por minuto, dorme e tenta de novo
                    print("-> Acelerei demais. Pausando 60 segundos por segurança...")
                    time.sleep(60)
            else:
                print(f"[ERRO BIZARRO] Pulando este lote devido a: {e}")
                break

# EXECUÇÃO DOS LOTES
status = ""

# Processa primeiro as Notas 1
st_lote = 1
for lote in lotes_ruins:
    status = processar_lote_com_pausa(lote, "Erros Críticos (Nota 1)", st_lote, len(lotes_ruins))
    if status == "PARAR_TUDO": break
    st_lote += 1

# Se ainda tiver cota, processa as Notas 5
if status != "PARAR_TUDO":
    st_lote = 1
    for lote in lotes_bons:
        status = processar_lote_com_pausa(lote, "Excelência Clínica (Nota 5)", st_lote, len(lotes_bons))
        if status == "PARAR_TUDO": break
        st_lote += 1

print("\n[PROCESSO CONCLUÍDO] Auditoria em Lotes finalizada!")
print("-> O arquivo 'relatorio_insights_completo.md' contém agora a maior quantidade de insights que sua cota permitiu gerar.")