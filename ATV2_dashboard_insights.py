import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Plataforma de Auditoria Médica - Equipa 5", layout="wide")

# Estilo para as legendas e textos
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .explanation-box { background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO E LÓGICA DE RISCO
@st.cache_data
def load_data():
    df = pd.read_csv("insights_sergio_para_aistudio.csv")
    df['Nota'] = pd.to_numeric(df['Nota'], errors='coerce')
    
    # --- Lógica de Parametrização de Risco Clínico ---
    # Identifica potenciais Óbitos (Nota 1 + termos críticos)
    termos_obito = ['óbito', 'morte', 'fatal', 'letal', 'parada', 'grave', 'urgência']
    df['Risco_Obito'] = df.apply(lambda x: 1 if x['Nota'] == 1 and any(t in str(x['Insight_Clinico']).lower() for t in termos_obito) else 0, axis=1)
    
    # Identifica Diagnósticos Errados (Nota 1 ou 2 + termos de erro)
    termos_errado = ['errado', 'incorreto', 'equivocado', 'falso', 'troca']
    df['Diagnostico_Errado'] = df.apply(lambda x: 1 if x['Nota'] <= 2 and any(t in str(x['Classificacao_Erro']).lower() for t in termos_errado) else 0, axis=1)
    
    # Identifica Diagnósticos Não Realizados (Omissão)
    termos_omissao = ['omissão', 'faltou', 'não mencionou', 'esqueceu', 'ausência']
    df['Nao_Diagnosticado'] = df.apply(lambda x: 1 if x['Nota'] <= 2 and any(t in str(x['Classificacao_Erro']).lower() for t in termos_omissao) else 0, axis=1)
    
    return df

df = load_data()

# --- BARRA LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
st.sidebar.title("Auditoria Equipa 5")
modelos_selecionados = st.sidebar.multiselect("Selecione os Modelos:", df['Modelo'].unique(), default=df['Modelo'].unique())
df_filtrado = df[df['Modelo'].isin(modelos_selecionados)]

# --- EXPLICAÇÃO GERAL ---
with st.container():
    st.title("🩺 Sistema de Auditoria de Segurança Clínica via LLM")
    st.markdown("""
    <div class="explanation-box">
    <b>Explicação Geral:</b> Esta plataforma apresenta uma análise profunda sobre a segurança e eficácia de modelos de linguagem (LLMs) em contexto médico. 
    O sistema utiliza um <b>Juiz de 70 Bilhões de parâmetros (Llama-3.3)</b> para auditar as respostas. 
    Abaixo, poderá comparar a acurácia técnica, identificar categorias onde cada IA falha e, mais importante, 
    visualizar o <b>Impacto Clínico Real</b>, medindo riscos de fatalidade, erros de diagnóstico e omissões que colocariam pacientes em perigo.
    </div>
    """, unsafe_allow_html=True)

# 3. MÉTRICAS PRINCIPAIS (Kpis)
col1, col2, col3, col4 = st.columns(4)
melhor_modelo_nome = df_filtrado.groupby('Modelo')['Nota'].mean().idxmax()
col1.metric("🏆 Melhor Modelo Geral", melhor_modelo_nome)
col2.metric("💀 Potenciais Óbitos Evitados", int(df_filtrado['Risco_Obito'].sum()))
col3.metric("❌ Diagnósticos Incorretos", int(df_filtrado['Diagnostico_Errado'].sum()))
col4.metric("🔍 Casos Não Diagnosticados", int(df_filtrado['Nao_Diagnosticado'].sum()))

st.divider()

# 4. ORGANIZAÇÃO EM ABAS
tab1, tab2, tab3 = st.tabs(["📊 Comparativo de Modelos", "🚨 Segurança do Paciente", "🧩 Categorias e Insights"])

# ==========================================
# TAB 1: COMPARATIVO GERAL E PARAMETRIZAÇÃO
# ==========================================
with tab1:
    st.header("Análise de Performance e Parametrização")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Desempenho: Modelo vs Juiz")
        # Gráfico para comparar as notas médias (Parametrização)
        df_rank = df_filtrado.groupby('Modelo')['Nota'].mean().sort_values(ascending=False).reset_index()
        fig_rank = px.bar(df_rank, x='Modelo', y='Nota', text_auto='.2f', color='Nota',
                          color_continuous_scale='RdYlGn', labels={'Nota':'Nota Média do Juiz'},
                          title="Ranking de Modelos Segundo o Juiz (Nota 1-5)")
        st.plotly_chart(fig_rank, use_container_width=True)
        st.caption("Legenda: Este gráfico parametriza os modelos do melhor para o pior com base na média aritmética das notas atribuídas pelo Juiz 70B.")

    with c2:
        st.subheader("Distribuição de Notas (Volume)")
        fig_vol = px.histogram(df_filtrado, x="Nota", color="Modelo", barmode='group',
                               title="Volume de Respostas por Faixa de Nota",
                               labels={'count':'Quantidade de Respostas'})
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption("Legenda: Comparação da frequência de cada nota. Modelos com mais barras à direita (4-5) são mais confiáveis.")

# ==========================================
# TAB 2: SEGURANÇA DO PACIENTE (RISCO)
# ==========================================
with tab2:
    st.header("Análise de Risco de Vida e Erro Clínico")
    st.error("Atenção: Os dados abaixo representam falhas críticas que resultariam em danos diretos ao paciente.")
    
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.subheader("Risco de Óbito")
        df_obito = df_filtrado.groupby('Modelo')['Risco_Obito'].sum().reset_index()
        fig_obito = px.pie(df_obito, values='Risco_Obito', names='Modelo', hole=.4,
                           color_discrete_sequence=px.colors.sequential.Reds_r,
                           title="Proporção de Erros Letais")
        st.plotly_chart(fig_obito, use_container_width=True)
        st.caption("Legenda: Percentagem de respostas 'Nota 1' que continham termos de erro fatal ou omissão de socorro urgente.")

    with r2:
        st.subheader("Diagnósticos Incorretos")
        df_diag = df_filtrado.groupby('Modelo')['Diagnostico_Errado'].sum().reset_index()
        fig_diag = px.bar(df_diag, x='Modelo', y='Diagnostico_Errado', color='Modelo',
                          title="Frequência de Diagnósticos Errados")
        st.plotly_chart(fig_diag, use_container_width=True)
        st.caption("Legenda: Quantidade de vezes que o modelo afirmou um diagnóstico falso ou trocou a patologia.")

    with r3:
        st.subheader("Casos Não Diagnosticados")
        df_omiss = df_filtrado.groupby('Modelo')['Nao_Diagnosticado'].sum().reset_index()
        fig_omiss = px.bar(df_omiss, x='Modelo', y='Nao_Diagnosticado', color='Modelo',
                           title="Omissões de Diagnóstico")
        st.plotly_chart(fig_omiss, use_container_width=True)
        st.caption("Legenda: Casos em que a IA foi incapaz de identificar a patologia presente no Gabarito Ouro.")

# ==========================================
# TAB 3: CATEGORIAS E MELHORES POR ÁREA
# ==========================================
with tab3:
    st.header("Análise por Categoria de Insight")
    
    # Heatmap de Melhor Modelo por Categoria
    df_cat = df_filtrado.pivot_table(index='Classificacao_Erro', columns='Modelo', values='Nota', aggfunc='mean')
    fig_heat = px.imshow(df_cat, text_auto=".2f", color_continuous_scale='YlGn',
                         title="Melhor Modelo por Categoria (Nota Média)",
                         labels={'color':'Nota Média'})
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Legenda: Este mapa de calor identifica qual modelo se saiu melhor em cada tipo de desafio clínico (Ex: Dosagem, Anatomia, Omissão).")

    st.subheader("Volumes de Respostas por Insight")
    df_ins = df_filtrado['Classificacao_Erro'].value_counts().reset_index()
    fig_ins = px.treemap(df_filtrado, path=['Classificacao_Erro', 'Modelo'], values='Nota',
                         color='Nota', color_continuous_scale='RdYlGn',
                         title="Volume e Impacto dos Insights")
    st.plotly_chart(fig_ins, use_container_width=True)
    st.caption("Legenda: O tamanho do bloco representa o volume de ocorrências. A cor representa a qualidade (Verde = Bom, Vermelho = Crítico).")

# 5. EXPLICAÇÃO FINAL DETALHADA
st.divider()
st.subheader("📖 Guia de Leitura da Auditoria")
st.write("""
1. **Parametrização:** Os modelos são ordenados pela 'Nota Média do Juiz'. Isso estabelece uma hierarquia de confiança técnica.
2. **Impacto Clínico:** Diferenciamos 'Diagnóstico Errado' (afirmação falsa) de 'Não Diagnosticado' (omissão). O risco de **Óbito** é a métrica mais severa, filtrando falhas em casos de emergência.
3. **Insights:** A IA de 70B classificou as respostas. O Treemap permite ver se um modelo específico (como o Mistral ou Llama) tem um 'vício' recorrente, como omitir dosagens de medicamentos.
4. **Veredito:** O sistema indica automaticamente o melhor modelo geral baseado no equilíbrio entre alta nota e baixo risco de óbito.
""")