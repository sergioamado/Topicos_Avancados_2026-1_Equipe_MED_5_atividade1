import os
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from dotenv import load_dotenv, find_dotenv

# CONFIGURAÇÕES INICIAIS
load_dotenv(find_dotenv())

print("-" * 50)
print("📊 ANÁLISE ESTATÍSTICA E DESEMPENHO DO RAG (EQUIPE 5)")
print("-" * 50)

# CONEXÃO COM O BANCO DE DADOS POSTGRESQL
try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    cursor = conn.cursor()
    print("[OK] Conectado ao PostgreSQL.")
except Exception as e:
    print(f"[ERRO] Falha no banco de dados: {e}")
    exit()

# SELEÇÃO DINÂMICA DE MEMBRO
cursor.execute("SELECT id_membro, nome FROM membros ORDER BY id_membro")
membros = cursor.fetchall()

if not membros:
    print("[ERRO] Nenhum membro encontrado.")
    exit()

print("\n👤 Selecione o membro para gerar o relatório:")
for m in membros:
    print(f"[{m[0]}] - {m[1].title()}")

while True:
    try:
        id_membro = int(input("\nDigite o ID do membro: "))
        if id_membro in [m[0] for m in membros]:
            nome_membro = next(m[1] for m in membros if m[0] == id_membro)
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Digite apenas números.")

# SELEÇÃO DO MODELO LOCAL AVALIADO
cursor.execute("SELECT DISTINCT modelo FROM atividade3_avaliacoes_juiz WHERE id_membro = %s", (id_membro,))
modelos_disponiveis = cursor.fetchall()

if not modelos_disponiveis:
    print("\n[AVISO] Nenhuma avaliação concluída encontrada para este membro.")
    exit()

print("\n🤖 Qual modelo você deseja analisar?")
for i, mod in enumerate(modelos_disponiveis):
    print(f"[{i}] - {mod[0]}")

while True:
    try:
        escolha_mod = int(input("\nDigite o número correspondente: "))
        if 0 <= escolha_mod < len(modelos_disponiveis):
            modelo_avaliado = modelos_disponiveis[escolha_mod][0]
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Digite apenas números.")

# SELEÇÃO DO TIPO DE QUESTÃO PARA ANÁLISE SEPARADA
print("\n📚 Qual banco de questões deseja analisar agora?")
print("[1] - Questões Abertas (Open - M1)")
print("[2] - Múltipla Escolha (MCQ - M2)")

while True:
    try:
        escolha_tipo = int(input("\nDigite 1 ou 2: "))
        if escolha_tipo in [1, 2]:
            tipo_questao = "Abertas" if escolha_tipo == 1 else "MCQ"
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Por favor, digite 1 ou 2.")

# EXTRAÇÃO DE DADOS (COM FILTRO RÍGOROSO POR TIPO DE QUESTÃO)
if escolha_tipo == 1:
    consulta_sql = """
        SELECT 
            a3.id_pergunta_original,
            COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
            a3.nota_juiz AS nota_com_rag
        FROM atividade3_avaliacoes_juiz a3
        INNER JOIN atividade1_open a1 
            ON a3.id_pergunta_original = a1.id_pergunta_original 
            AND a3.id_membro = a1.id_membro 
            AND a3.modelo = a1.modelo
        LEFT JOIN atividade2_insights a2 
            ON a3.id_pergunta_original = a2.id_pergunta 
            AND a3.id_membro = a2.id_membro 
            AND a3.modelo = a2.modelo
        WHERE a3.id_membro = %s AND a3.modelo = %s
        ORDER BY a3.id_pergunta_original;
    """
else:
    consulta_sql = """
        SELECT 
            a3.id_pergunta_original,
            COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
            a3.nota_juiz AS nota_com_rag
        FROM atividade3_avaliacoes_juiz a3
        INNER JOIN atividade1_mcq a1 
            ON a3.id_pergunta_original = CAST(a1.id_mcq AS VARCHAR) 
            AND a3.id_membro = a1.id_membro 
            AND a3.modelo = a1.modelo
        LEFT JOIN atividade2_insights a2 
            ON a3.id_pergunta_original = a2.id_pergunta 
            AND a3.id_membro = a2.id_membro 
            AND a3.modelo = a2.modelo
        WHERE a3.id_membro = %s AND a3.modelo = %s
        ORDER BY CAST(a3.id_pergunta_original AS INTEGER);
    """

cursor.execute(consulta_sql, (id_membro, modelo_avaliado))
dados = cursor.fetchall()

if not dados:
    print(f"\n[ERRO] Não foram encontrados dados avaliados para as Questões {tipo_questao} neste modelo.")
    exit()

# CONVERSÃO PARA DATAFRAME PANDAS
df = pd.DataFrame(dados, columns=['id_pergunta', 'nota_sem_rag', 'nota_com_rag'])

# CÁLCULOS ESTATÍSTICOS
total_questoes = len(df)
media_sem_rag = df['nota_sem_rag'].mean()
media_com_rag = df['nota_com_rag'].mean()
ganho_absoluto = media_com_rag - media_sem_rag
percentual_melhoria = (ganho_absoluto / media_sem_rag) * 100 if media_sem_rag > 0 else 0

# Coeficiente de Correlação de Spearman (com proteção contra variância zero)
if df['nota_sem_rag'].nunique() > 1 and df['nota_com_rag'].nunique() > 1:
    coef_spearman, p_valor = spearmanr(df['nota_sem_rag'], df['nota_com_rag'])
else:
    coef_spearman, p_valor = (0.0, 1.0) # Assume 0 se todas as notas forem iguais e não houver variação

# IMPRESSÃO DO RELATÓRIO NO TERMINAL
print("\n" + "="*60)
print(f"📈 RELATÓRIO FINAL: {modelo_avaliado.upper()} | TIPO: {tipo_questao.upper()}")
print("="*60)
print(f"👨‍⚕️ Membro responsável : {nome_membro.title()}")
print(f"📝 Total de questões  : {total_questoes}")
print("-" * 60)
print(f"❌ Média ANTES do RAG : {media_sem_rag:.2f} / 5.0")
print(f"✅ Média DEPOIS do RAG: {media_com_rag:.2f} / 5.0")
print(f"🚀 Ganho de Desempenho: +{ganho_absoluto:.2f} pontos ({percentual_melhoria:.1f}%)")
print("-" * 60)

if not np.isnan(coef_spearman):
    print(f"📊 Correlação de Spearman (ρ): {coef_spearman:.3f}")
    print(f"P-Valor: {p_valor:.4f}")
    
    if coef_spearman > 0.7:
        print("💡 Interpretação: Correlação FORTE. O modelo manteve a consistência, mas com notas mais altas.")
    elif coef_spearman > 0.4:
        print("💡 Interpretação: Correlação MODERADA. O RAG alterou o padrão de respostas de forma considerável.")
    else:
        print("💡 Interpretação: Correlação FRACA. O RAG alterou totalmente as respostas que a IA daria de memória.")
else:
    print("📊 Correlação de Spearman: Não aplicável (falta de variância nos dados).")
print("="*60)

# GERAÇÃO DO GRÁFICO VISUAL (SALVO EM PNG)
plt.figure(figsize=(12, 6))
largura_barra = 0.35
indices = np.arange(total_questoes)

plt.bar(indices - largura_barra/2, df['nota_sem_rag'], width=largura_barra, label='Sem RAG (Memória Pura)', color='#e74c3c')
plt.bar(indices + largura_barra/2, df['nota_com_rag'], width=largura_barra, label='Com RAG (PDF ADA/AHA)', color='#2ecc71')

plt.xlabel('ID da Pergunta', fontweight='bold')
plt.ylabel('Nota do Juiz IA (1 a 5)', fontweight='bold')
plt.title(f'Impacto do RAG no Raciocínio Clínico - {modelo_avaliado} ({tipo_questao})', fontweight='bold')
plt.xticks(indices, df['id_pergunta'], rotation=45 if total_questoes > 10 else 0)
plt.ylim(0, 5.5)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Salvando o gráfico com o tipo de questão no nome do ficheiro
nome_arquivo = f"grafico_rag_{tipo_questao}_{nome_membro.replace(' ', '_')}_{modelo_avaliado}.png"
plt.savefig(nome_arquivo, dpi=300)
print(f"\n[SUCESSO] Gráfico comparativo gerado e guardado como: {nome_arquivo}")

cursor.close()
conn.close()