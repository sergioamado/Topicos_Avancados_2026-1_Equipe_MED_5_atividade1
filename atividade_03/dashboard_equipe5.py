import os
import pandas as pd
import psycopg2
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from dotenv import load_dotenv, find_dotenv

st.set_page_config(
    page_title="Dashboard RAG - Equipe 5",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

@st.cache_resource
def init_connection():
    """Inicializa a ligação ao PostgreSQL"""
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
    except Exception as e:
        st.error(f"Falha ao conectar ao banco de dados: {e}")
        return None

conn = init_connection()

@st.cache_data
def get_membros():
    df = pd.read_sql("SELECT id_membro, nome FROM membros ORDER BY id_membro", conn)
    return df

@st.cache_data
def get_modelos(id_membro):
    query = """
        SELECT DISTINCT modelo FROM atividade3_avaliacoes_juiz WHERE id_membro = %s
        UNION 
        SELECT DISTINCT modelo FROM atividade1_open WHERE id_membro = %s
    """
    df = pd.read_sql(query, conn, params=(id_membro, id_membro))
    return df['modelo'].tolist()

@st.cache_data
def get_dados_atividade1_open(id_membro, modelo):
    query = "SELECT id_pergunta_original, pergunta, resposta_ia, gabarito_esperado FROM atividade1_open WHERE id_membro = %s AND modelo = %s"
    return pd.read_sql(query, conn, params=(id_membro, modelo))

@st.cache_data
def get_dados_atividade2(id_membro, modelo):
    query = "SELECT id_pergunta, nota_juiz, classificacao_erro, insight_clinico, justificativa_juiz FROM atividade2_insights WHERE id_membro = %s AND modelo = %s"
    return pd.read_sql(query, conn, params=(id_membro, modelo))

@st.cache_data
def get_dados_comparativos(id_membro, modelo, tipo_questao="Open"):
    if tipo_questao == "Open":
        query = """
            SELECT 
                a3.id_pergunta_original,
                COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
                a3.nota_juiz AS nota_com_rag,
                r3.pergunta,
                r3.resposta_ia_com_rag,
                a3.chain_of_thought AS justificativa_nova
            FROM atividade3_avaliacoes_juiz a3
            JOIN atividade3_respostas_rag r3 ON a3.id_resposta_rag = r3.id_resposta_rag
            INNER JOIN atividade1_open a1 ON a3.id_pergunta_original = a1.id_pergunta_original AND a3.id_membro = a1.id_membro AND a3.modelo = a1.modelo
            LEFT JOIN atividade2_insights a2 ON a3.id_pergunta_original = a2.id_pergunta AND a3.id_membro = a2.id_membro AND a3.modelo = a2.modelo
            WHERE a3.id_membro = %s AND a3.modelo = %s
            ORDER BY a3.id_pergunta_original;
        """
    else:
        query = """
            SELECT 
                a3.id_pergunta_original,
                COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
                a3.nota_juiz AS nota_com_rag,
                r3.pergunta,
                r3.resposta_ia_com_rag,
                a3.chain_of_thought AS justificativa_nova
            FROM atividade3_avaliacoes_juiz a3
            JOIN atividade3_respostas_rag r3 ON a3.id_resposta_rag = r3.id_resposta_rag
            INNER JOIN atividade1_mcq a1 ON a3.id_pergunta_original = CAST(a1.id_mcq AS VARCHAR) AND a3.id_membro = a1.id_membro AND a3.modelo = a1.modelo
            LEFT JOIN atividade2_insights a2 ON a3.id_pergunta_original = a2.id_pergunta AND a3.id_membro = a2.id_membro AND a3.modelo = a2.modelo
            WHERE a3.id_membro = %s AND a3.modelo = %s
            ORDER BY CAST(a3.id_pergunta_original AS INTEGER);
        """
    return pd.read_sql(query, conn, params=(id_membro, modelo))


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Configurações")
st.sidebar.markdown("Selecione os filtros abaixo para atualizar o dashboard.")

membros_df = get_membros()
if membros_df.empty:
    st.error("Nenhum membro encontrado na base de dados.")
    st.stop()

membro_selecionado = st.sidebar.selectbox(
    "👤 Selecione o Membro:", 
    membros_df['nome'].str.title().tolist()
)
id_membro = int(membros_df.loc[membros_df['nome'].str.title() == membro_selecionado, 'id_membro'].values[0])

modelos_disponiveis = get_modelos(id_membro)
if not modelos_disponiveis:
    st.sidebar.warning("Nenhum modelo encontrado para este membro.")
    st.stop()

modelo_selecionado = st.sidebar.selectbox("🤖 Selecione o Modelo Local:", modelos_disponiveis)

tipo_questao_selecionada = st.sidebar.radio("📚 Tipo de Questão para Análise:", ["Questões Abertas (Open)", "Múltipla Escolha (MCQ)"])
tipo_filtro = "Open" if "Open" in tipo_questao_selecionada else "MCQ"

st.sidebar.markdown("---")
st.sidebar.info("Projeto: Tópicos Avançados\n\nEquipe: MED 5")


st.title("🏥 Dashboard MLOps - Inferência Clínica e RAG")
st.markdown(f"**Analisando dados de:** `{membro_selecionado}` | **Modelo em foco:** `{modelo_selecionado}`")
st.markdown("Este painel consolida os resultados das 3 fases do projeto, mostrando a evolução da Inteligência Artificial em responder a questões médicas complexas antes e depois da injeção de conhecimento externo (RAG).")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Visão Geral", 
    "📝 Ativ. 1: Memória do Modelo", 
    "⚖️ Ativ. 2: O Juiz IA", 
    "🚀 Ativ. 3: Impacto do RAG"
])

with tab1:
    st.header("Pipeline do Projeto")
    
    col1, col2, col3 = st.columns(3)
    col1.info("**Atividade 1 (Sem RAG):**\nO modelo local responde a questões clínicas usando apenas o que aprendeu no seu treino base. As respostas tendem a ter alucinações.")
    col2.warning("**Atividade 2 (Juiz Nuvem):**\nUm LLM gigantesco (Gemini/Groq) atua como juiz. Ele lê a resposta do modelo local, compara com o gabarito oficial e dá uma nota de 1 a 5.")
    col3.success("**Atividade 3 (Com RAG):**\nA IA local recebe fragmentos dos PDFs da ADA e AHA (via ChromaDB) como 'dica' antes de responder. O Juiz avalia novamente.")
    
    st.markdown("### Resumo da Base de Dados")
    df_comparativo = get_dados_comparativos(id_membro, modelo_selecionado, tipo_filtro)
    
    if not df_comparativo.empty:
        met1, met2, met3 = st.columns(3)
        met1.metric("Questões Analisadas", len(df_comparativo))
        
        media_sem = df_comparativo['nota_sem_rag'].mean()
        media_com = df_comparativo['nota_com_rag'].mean()
        evolucao = media_com - media_sem
        
        met2.metric("Nota Média (Sem RAG)", f"{media_sem:.2f} / 5.0")
        met3.metric("Nota Média (Com RAG)", f"{media_com:.2f} / 5.0", f"{evolucao:+.2f} pontos")
    else:
        st.warning("Não há dados consolidados suficientes para exibir as métricas gerais.")


with tab2:
    st.header("Atividade 1: Respostas por Conhecimento Base")
    st.markdown("Aqui podemos observar as respostas puras geradas pela IA sem qualquer ajuda externa.")
    
    if tipo_filtro == "Open":
        df_ativ1 = get_dados_atividade1_open(id_membro, modelo_selecionado)
        if not df_ativ1.empty:
            st.dataframe(df_ativ1, use_container_width=True, hide_index=True)
            
            st.markdown("### Detalhamento de uma Questão")
            questao_detalhe = st.selectbox("Escolha uma questão para ler os detalhes:", df_ativ1['id_pergunta_original'].tolist())
            linha = df_ativ1[df_ativ1['id_pergunta_original'] == questao_detalhe].iloc[0]
            
            with st.expander("Ver Pergunta Completa", expanded=True):
                st.write(linha['pergunta'])
            colA, colB = st.columns(2)
            with colA:
                st.success("**Gabarito Oficial:**")
                st.write(linha['gabarito_esperado'])
            with colB:
                st.error(f"**Resposta da IA ({modelo_selecionado}):**")
                st.write(linha['resposta_ia'])
        else:
            st.info("Não há dados da Atividade 1 (Open) para este filtro.")
    else:
        st.info("Para simplificar o dashboard, focamos os textos longos nas questões Abertas. Altere para 'Open' no menu lateral para ver os textos gerados de memória.")

with tab3:
    st.header("Atividade 2: Avaliação Crítica do Juiz")
    df_ativ2 = get_dados_atividade2(id_membro, modelo_selecionado)
    
    if not df_ativ2.empty:
        colX, colY = st.columns([1, 2])
        
        with colX:
            st.subheader("Distribuição de Notas (Sem RAG)")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df_ativ2, x='nota_juiz', palette="Reds_r", ax=ax)
            ax.set_xlabel("Nota do Juiz (1 a 5)")
            ax.set_ylabel("Quantidade de Questões")
            st.pyplot(fig)
            
        with colY:
            st.subheader("Tipos de Erro Cometidos (Sem RAG)")
            erros = df_ativ2['classificacao_erro'].value_counts()
            st.bar_chart(erros)
            
        st.markdown("### Insights Clínicos Gerados pelo Juiz")
        st.dataframe(df_ativ2[['id_pergunta', 'nota_juiz', 'justificativa_juiz', 'insight_clinico']], use_container_width=True)
    else:
        st.warning("Não há avaliações da Atividade 2 para este modelo/membro.")


with tab4:
    st.header("Atividade 3: A Cura das Alucinações com RAG")
    
    df_comp = get_dados_comparativos(id_membro, modelo_selecionado, tipo_filtro)
    
    if df_comp.empty:
        st.warning("Faltam dados da Atividade 3 para gerar os comparativos finais.")
    else:
        # CÁLCULO DE SPEARMAN
        if df_comp['nota_sem_rag'].nunique() > 1 and df_comp['nota_com_rag'].nunique() > 1:
            coef_spearman, p_valor = spearmanr(df_comp['nota_sem_rag'], df_comp['nota_com_rag'])
        else:
            coef_spearman, p_valor = (0.0, 1.0)
            
        st.subheader("Estatísticas Avançadas")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Correlação de Spearman", f"{coef_spearman:.3f}")
        sc2.metric("P-Valor (Significância)", f"{p_valor:.4f}")
        
        if coef_spearman > 0.7:
            interpretacao = "FORTE: O modelo manteve o padrão mas subiu de nível."
        elif coef_spearman > 0.4:
            interpretacao = "MODERADA: O RAG mudou o comportamento do modelo consideravelmente."
        else:
            interpretacao = "FRACA: O RAG revolucionou por completo as respostas anteriores."
        sc3.info(interpretacao)

        st.markdown("---")
        
        # GRÁFICO COMPARATIVO (O MAIS IMPORTANTE)
        st.subheader("Comparativo de Desempenho: Antes vs Depois do RAG")
        
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        largura = 0.35
        x = range(len(df_comp))
        
        ax2.bar([i - largura/2 for i in x], df_comp['nota_sem_rag'], width=largura, label='Sem RAG', color='#e74c3c')
        ax2.bar([i + largura/2 for i in x], df_comp['nota_com_rag'], width=largura, label='Com RAG (AHA/ADA)', color='#2ecc71')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_comp['id_pergunta_original'], rotation=45 if len(x) > 10 else 0)
        ax2.set_ylim(0, 5.5)
        ax2.set_ylabel("Nota do Juiz")
        ax2.set_xlabel("ID da Questão")
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        st.pyplot(fig2)
        
        st.markdown("---")
        st.subheader("🕵️‍♂️ Inspeção a Fundo (Como o Juiz Avaliou o RAG)")
        
        id_selecionado = st.selectbox("Selecione uma questão para ver como o RAG mudou a resposta:", df_comp['id_pergunta_original'])
        detalhe_rag = df_comp[df_comp['id_pergunta_original'] == id_selecionado].iloc[0]
        
        st.write(f"**Nota Antiga:** {detalhe_rag['nota_sem_rag']} ➔ **Nota Nova (RAG):** {detalhe_rag['nota_com_rag']}")
        
        with st.expander("Ler Justificativa Final do Juiz", expanded=True):
            st.info(detalhe_rag['justificativa_nova'])
            
        with st.expander("Ler a Nova Resposta Gerada pelo LLM"):
            st.write(detalhe_rag['resposta_ia_com_rag'])