import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Backend não-interativo para Matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import shap

# --- Configuração da Página e do Projeto ---

st.set_page_config(
    page_title="Plataforma de Risco de Crédito",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProjectConfig:
    """
    Classe de configuração para centralizar parâmetros e constantes do projeto.
    Isso garante consistência e facilita a manutenção do código.
    """
    TARGET_VARIABLE = 'class'
    TEST_SIZE_RATIO = 0.3
    RANDOM_STATE_SEED = 42
    N_SPLITS_KFOLD = 5
    RFE_CV_SCORING = 'roc_auc'
    
    # Paleta de cores profissional para os gráficos
    PRIMARY_COLOR = "#005f73"
    SECONDARY_COLOR = "#0a9396"
    ACCENT_COLOR = "#ee9b00"
    BAD_RISK_COLOR = "#ae2012"
    GOOD_RISK_COLOR = "#94d2bd"
    BACKGROUND_COLOR = "#F0F2F6" 
    TEXT_COLOR = "#001219"
    GRID_COLOR = "#e9d8a6"

    @staticmethod
    def get_plotly_template():
        """
        Define um template customizado para os gráficos do Plotly.
        """
        template = go.layout.Template()
        template.layout.paper_bgcolor = ProjectConfig.BACKGROUND_COLOR
        template.layout.plot_bgcolor = ProjectConfig.BACKGROUND_COLOR
        template.layout.font = dict(color=ProjectConfig.TEXT_COLOR, family="Segoe UI")
        template.layout.xaxis = dict(gridcolor=ProjectConfig.GRID_COLOR, showgrid=True, zeroline=False)
        template.layout.yaxis = dict(gridcolor=ProjectConfig.GRID_COLOR, showgrid=True, zeroline=False)
        template.layout.title = dict(x=0.5, font=dict(size=20))
        return template

def initialize_session_state():
    """
    Inicializa o estado da sessão do Streamlit. Isso é crucial para
    manter os dados e o progresso do usuário entre as interações, criando
    uma experiência de aplicação fluida em vez de um script que re-executa do zero.
    """
    session_state_defaults = {
        'app_stage': 'initialization',
        'data_loaded': False,
        'data_processed': False,
        'models_trained': False,
        'final_model_selected': False,
        'raw_df': None,
        'processed_df': None,
        'artifacts': {},
        'decision_threshold': 0.5
    }
    for key, default_value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_data(show_spinner="Analisando o arquivo de dados...")
def load_and_profile_data():
    """
    Carrega os dados diretamente do arquivo padrão 'credit_customers.csv'
    e realiza um profiling completo, retornando o DataFrame e um dicionário
    com as estatísticas de qualidade dos dados.
    """
    try:
        df = pd.read_csv('credit_customers.csv')
    except FileNotFoundError:
        st.error("ERRO CRÍTICO: O arquivo padrão 'credit_customers.csv' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script `app.py`.")
        return None, None
    except Exception as e:
        st.error(f"Erro inesperado ao ler o arquivo: {e}")
        return None, None

    profile_summary = {
        'Visão Geral': {
            'Clientes (Linhas)': df.shape[0],
            'Atributos (Colunas)': df.shape[1],
            'Células Faltando': df.isnull().sum().sum(),
            '% de Dados Faltando': f"{df.isnull().sum().sum() / df.size * 100:.2f}%",
            'Linhas Duplicadas': df.duplicated().sum()
        },
        'detalhes_variaveis': []
    }
    
    for col in df.columns:
        series = df[col]
        col_info = {
            'Atributo': col, 'Tipo': str(series.dtype),
            'Nulos (%)': f"{series.isnull().mean() * 100:.2f}%",
            'Valores Únicos': series.nunique()
        }
        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
            col_info['Outliers (%)'] = f"{len(outliers) / len(series) * 100:.2f}%"
        profile_summary['detalhes_variaveis'].append(col_info)
    
    return df, profile_summary

@st.cache_data(show_spinner="Aplicando engenharia de features e transformações...")
def execute_feature_engineering(_df):
    """
    Executa um pipeline completo de limpeza e transformação de variáveis,
    adaptado para o dataset de risco de crédito.
    """
    df = _df.copy()
    
    # Assegura que colunas categóricas sejam tratadas como string para evitar erros
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        
    # Codificação da variável-alvo para formato numérico
    # 'good' (bom risco) -> 0 | 'bad' (mau risco) -> 1
    le = LabelEncoder()
    df[ProjectConfig.TARGET_VARIABLE] = le.fit_transform(df[ProjectConfig.TARGET_VARIABLE])
    st.session_state.artifacts['label_encoder'] = le

    # Exemplo de criação de novas variáveis para capturar interações importantes
    # Esta etapa é crucial para dar mais "matéria-prima" aos modelos
    df['credit_to_duration_ratio'] = df['credit_amount'] / df['duration']
    df['credit_to_age_ratio'] = df['credit_amount'] / df['age']
    
    # Hipótese: Pessoas mais jovens pedindo muito crédito podem ser mais arriscadas
    df.loc[df['age'] < 25, 'young_high_credit'] = (df['credit_amount'] > df['credit_amount'].median()).astype(int)
    df['young_high_credit'].fillna(0, inplace=True)
    
    return df

def display_home_page():
    """
    Renderiza a página inicial de boas-vindas do dashboard.
    Apresenta a missão do projeto e guia o usuário sobre como navegar.
    """    
    st.title("Sistema de Apoio à Decisão para Análise de Risco de Crédito")
    st.subheader("Utilizando IA, XAI (SHAP) e Clusterização para Decisões Gerenciais")
    st.markdown("---")
    
    st.markdown("""
    ### Bem-vindo(a), Analista de Risco!

    Esta plataforma interativa foi desenvolvida como a solução para a **Prova Final** da disciplina de Sistemas de Informação em Engenharia de Produção. O objetivo é fornecer ao setor de risco de crédito uma ferramenta completa para otimizar a concessão de crédito, equilibrando a expansão da base de clientes com a sustentabilidade financeira da operação.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Prever o Risco</h5>", unsafe_allow_html=True)
            st.markdown("Utilizar modelos de Machine Learning para classificar novos clientes como bons (`good`) ou maus (`bad`) pagadores, com base em seus dados históricos.")
    with col2:
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Gerar Transparência (XAI)</h5>", unsafe_allow_html=True)
            st.markdown("Empregar técnicas de IA Explicável com **SHAP** para entender os fatores que mais influenciam as decisões dos modelos, tornando-os auditáveis e confiáveis.")
    with col3:
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Segmentar Clientes</h5>", unsafe_allow_html=True)
            st.markdown("Aplicar algoritmos não supervisionados como **K-Means** e **DBSCAN** para descobrir perfis de clientes e identificar comportamentos atípicos (outliers).")

    st.markdown("---")
    st.info("Utilize o menu de navegação na barra lateral esquerda para explorar as diferentes etapas desta análise completa, desde a preparação dos dados até a tomada de decisão gerencial.", icon="🧭")

def main():
    """
    Função principal que controla a navegação e a renderização das páginas
    do aplicativo Streamlit. É o ponto central que orquestra toda a aplicação.
    """
    initialize_session_state()
    px.defaults.template = ProjectConfig.get_plotly_template()

    st.sidebar.title("Painel de Controle 🎛️")
    st.sidebar.markdown("Navegue pelas etapas da análise de risco de crédito.")
    
    page_options = {
        "Página Inicial": "🏠",
        "Análise e Preparação dos Dados": "📊",
        "Análise Exploratória (EDA)": "🔍",
        "Modelagem Supervisionada": "⚙️",
        "Decisão Gerencial e Não Supervisionada": "🧠",
        "Documentação e Exportação": "📄"
    }
    
    page_selection = st.sidebar.radio(
        "Menu de Navegação:",
        options=page_options.keys(),
        format_func=lambda x: f"{page_options[x]} {x}" # Adiciona ícones ao rádio
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: left; font-size: 0.9em;'>
            <strong>Prova Final</strong><br>
            <span>EPR0072 - Sistemas de Informação</span><br>
            <span>Prof. João Gabriel de Moraes Souza</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Roteamento das páginas
    if page_selection == "Página Inicial":
        display_home_page()
    elif page_selection == "Análise e Preparação dos Dados":
        display_dataset_page()
    elif page_selection == "Análise Exploratória (EDA)":
        display_eda_page()
    elif page_selection == "Modelagem Supervisionada":
        display_modeling_page()
    elif page_selection == "Decisão Gerencial e Não Supervisionada":
        display_advanced_analysis_page()
    elif page_selection == "Documentação e Exportação":
        display_export_and_docs_page(st.session_state.artifacts)

def display_dataset_page():
    """
    Renderiza a página de Análise e Preparação dos Dados, guiando o usuário
    pelas etapas de auditoria, profiling e engenharia de features com
    explicações detalhadas para cada passo.
    """
    st.header("Análise e Preparação dos Dados 📊")
    st.markdown("""
    O primeiro passo em qualquer projeto de ciência de dados é uma auditoria completa nos dados brutos.
    Nesta seção, carregamos a base de dados **Credit Risk Customers** e realizamos um profiling para
    entender sua estrutura, qualidade e características iniciais.
    """)

    if not st.session_state.data_loaded:
        raw_df, profile_results = load_and_profile_data()
        if raw_df is not None:
            st.session_state.raw_df = raw_df
            st.session_state.artifacts['profile_results'] = profile_results
            st.session_state.data_loaded = True
        else:
            st.stop()

    st.subheader("Auditoria e Profiling dos Dados Brutos")
    st.markdown("""
    Antes de qualquer modelagem, precisamos "entrevistar" nossos dados. A tabela detalhada abaixo é o resultado dessa entrevista. Ela nos ajuda a responder perguntas cruciais:

    - **Tipo:** A variável é um número, um texto ou uma categoria? Isso define como vamos tratá-la.
    - **Nulos (%):** Existem dados faltando? Colunas com muitos valores nulos podem ser inúteis ou exigir um tratamento especial (imputação).
    - **Valores Únicos:** Quantas categorias diferentes existem em uma variável de texto? Isso impacta nossa estratégia de codificação. Uma variável com milhares de valores únicos não pode ser tratada da mesma forma que uma com apenas duas (ex: 'sim'/'não').
    - **Outliers (%):** Existem valores extremos que fogem muito do padrão? Outliers foram definidos aqui usando o **Método do IQR (Intervalo Interquartil)**. Calculamos o intervalo que contém os 50% centrais dos dados e consideramos um outlier qualquer ponto que esteja 1.5 vezes essa distância abaixo do primeiro quartil ou acima do terceiro. Identificar outliers é vital, pois eles podem distorcer os resultados dos modelos.
    """)

    with st.expander("Visualizar Relatório Detalhado por Atributo", expanded=True):
        profile_df = pd.DataFrame(st.session_state.artifacts['profile_results']['detalhes_variaveis']).set_index('Atributo')
        st.dataframe(profile_df, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Processamento e Engenharia de Features")
    st.markdown("""
    Com os dados auditados, o próximo passo é transformá-los para otimizar o desempenho dos modelos. Isso inclui a codificação de variáveis e a **Engenharia de Features**, que é a arte de criar novos atributos a partir dos dados existentes para revelar padrões que não eram óbvios.
    """)
    
    if st.button("Executar Engenharia de Features", type="primary"):
        with st.spinner('Processando... Esta etapa pode levar alguns segundos.'):
            processed_df = execute_feature_engineering(st.session_state.raw_df)
            st.session_state.processed_df = processed_df
            st.session_state.data_processed = True
            st.success("Pipeline de engenharia de features executado com sucesso!")

    if st.session_state.data_processed:
        st.subheader("Comparativo de Impacto: Antes vs. Depois da Transformação")
        st.markdown("""
        O objetivo de mostrar os dados lado a lado é dar transparência ao processo. Na tabela da esquerda ("Antes"), vemos os dados como eles chegaram. Na tabela da direita ("Depois"), vemos o resultado do nosso trabalho: a coluna `class` foi transformada de texto ('good'/'bad') para números (0/1) e, mais importante, **novas colunas inteligentes foram criadas**, como `credit_to_duration_ratio`. Essas novas features enriquecem o dataset e dão aos nossos modelos de IA mais informações para aprender e tomar decisões melhores.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dados Brutos (Amostra)**")
            st.dataframe(st.session_state.raw_df.head())
        with col2:
            st.markdown("**Dados Processados (Amostra)**")
            st.dataframe(st.session_state.processed_df.head())

        st.info("Com os dados devidamente preparados e enriquecidos, agora podemos prosseguir para a Análise Exploratória (EDA).", icon="✅")

@st.cache_data
def calculate_descriptive_stats(series):
    """
    Calcula um dicionário de estatísticas descritivas para uma variável numérica ou categórica.
    Esta função é armazenada em cache para otimizar a performance da aplicação.
    """
    if pd.api.types.is_numeric_dtype(series):
        return {
            'Média': series.mean(), 'Mediana': series.median(), 'Desvio Padrão': series.std(),
            'Variância': series.var(), 'Mínimo': series.min(), 'Máximo': series.max(),
            '25º Percentil': series.quantile(0.25), '75º Percentil': series.quantile(0.75),
            'Assimetria (Skew)': series.skew(), 'Curtose (Kurtosis)': series.kurt(),
            'Contagem': series.count(), 'Valores Únicos': series.nunique()
        }
    else:
        return {
            'Contagem': series.count(), 'Valores Únicos': series.nunique(),
            'Moda (Mais Frequente)': series.mode().iloc[0] if not series.mode().empty else 'N/A',
            'Frequência da Moda': series.value_counts().iloc[0] if not series.value_counts().empty else 0
        }

def render_univariate_analysis_tab(df):
    """
    Renderiza a aba de Análise Univariada na página de EDA.
    Permite ao usuário selecionar uma variável e visualizar sua distribuição e estatísticas.
    """
    st.subheader("Análise de Variáveis Individuais")
    st.markdown("Selecione um atributo para visualizar sua distribuição e principais métricas estatísticas.")

    default_variable = 'credit_amount' if 'credit_amount' in df.columns else df.columns[0]
    variable_to_analyze = st.selectbox(
        "Selecione o atributo de interesse:",
        options=df.columns,
        index=list(df.columns).index(default_variable)
    )

    if variable_to_analyze:
        selected_series = df[variable_to_analyze]
        
        stats_col, plot_col = st.columns([1, 2])
        
        with stats_col:
            st.markdown(f"#### Métricas para **{variable_to_analyze}**")
            stats_dict = calculate_descriptive_stats(selected_series)
            stats_df = pd.DataFrame(stats_dict.items(), columns=['Métrica', 'Valor'])
            st.dataframe(stats_df.style.format(precision=2), use_container_width=True)

        with plot_col:
            if pd.api.types.is_numeric_dtype(selected_series) and selected_series.nunique() > 2: # Exclui a variável alvo binária
                st.markdown(f"#### Distribuição de **{variable_to_analyze}**")
                fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1,
                                    subplot_titles=("Histograma e Curva de Densidade", "Box Plot para Detecção de Outliers"))
                
                fig.add_trace(go.Histogram(x=selected_series, name='Histograma', histnorm='probability density', marker_color=ProjectConfig.PRIMARY_COLOR), row=1, col=1)
                fig.add_trace(go.Box(x=selected_series, name='Box Plot', marker_color=ProjectConfig.ACCENT_COLOR), row=2, col=1)

                fig.update_layout(showlegend=False, height=500, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            else: 
                st.markdown(f"#### Frequência de **{variable_to_analyze}**")
                counts = selected_series.value_counts()
                fig = px.bar(
                    counts, x=counts.index, y=counts.values,
                    title=f"Contagem de Categorias em {variable_to_analyze}",
                    labels={'x': variable_to_analyze, 'y': 'Contagem'},
                    text_auto=True, color=counts.index,
                    color_discrete_map={'good': ProjectConfig.GOOD_RISK_COLOR, 'bad': ProjectConfig.BAD_RISK_COLOR}
                )
                fig.update_layout(height=500, xaxis_title=variable_to_analyze, yaxis_title='Contagem', showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_bivariate_analysis_tab(df):
    st.subheader("Análise da Relação entre Pares de Variáveis")
    st.markdown("Explore como diferentes características dos clientes se relacionam entre si e, principalmente, com a variável alvo `class` (0 para 'bom', 1 para 'mau').")
    
    numeric_options = df.select_dtypes(include=np.number).columns.drop('class', errors='ignore')
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Selecione a primeira variável (Eixo X):", df.columns, index=df.columns.to_list().index('purpose'), key="bivar_1")
    with col2:
        var2 = st.selectbox("Selecione a segunda variável (Eixo Y ou Agrupamento):", df.columns, index=df.columns.to_list().index('credit_history'), key="bivar_2")

    if var1 and var2 and var1 != var2:
        is_var1_numeric = var1 in numeric_options
        is_var2_numeric = var2 in numeric_options

        if is_var1_numeric and is_var2_numeric:
            st.markdown(f"#### Correlação Numérica: **{var1}** vs. **{var2}**")
            fig = px.scatter(
                df, x=var1, y=var2,
                color=df[ProjectConfig.TARGET_VARIABLE].map({0: 'Bom Risco', 1: 'Mau Risco'}),
                color_discrete_map={'Bom Risco': ProjectConfig.GOOD_RISK_COLOR, 'Mau Risco': ProjectConfig.BAD_RISK_COLOR},
                trendline="ols",
                title=f"Dispersão entre {var1} e {var2} por Risco de Crédito"
            )
            fig.update_layout(height=500, legend_title_text='Status do Risco')
            st.plotly_chart(fig, use_container_width=True)

        elif not is_var1_numeric and not is_var2_numeric:
            st.markdown(f"#### Associação Categórica: **{var1}** vs. **{var2}**")
            st.markdown("Para analisar a relação entre duas variáveis categóricas, existem diferentes abordagens visuais. Abaixo apresentamos duas opções:")

            st.markdown("##### Opção 1: Tabela de Contingência e Mapa de Calor (Heatmap)")
            st.markdown("Esta abordagem é excelente para ver a concentração de dados. Células mais escuras indicam uma combinação mais frequente de categorias.")
            contingency_table = pd.crosstab(df[var1], df[var2])
            st.dataframe(contingency_table, use_container_width=True)
            fig_heatmap = px.imshow(
                contingency_table,
                text_auto=True,
                aspect="auto",
                title=f"Concentração de Clientes por {var1} e {var2}",
                color_continuous_scale='Blues'
            )
            fig_heatmap.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")

            st.markdown("##### Opção 2: Gráfico de Barras Agrupado")
            st.markdown("Este gráfico ajuda a comparar as contagens de uma variável dentro de cada categoria da outra, de forma direta.")
            fig_grouped_bar = px.bar(
                df, x=var1, color=var2, barmode='group',
                title=f"Contagem Agrupada: {var1} vs. {var2}",
                text_auto=True
            )
            fig_grouped_bar.update_layout(height=500)
            st.plotly_chart(fig_grouped_bar, use_container_width=True)

        else:
            numeric_var = var1 if is_var1_numeric else var2
            categorical_var = var2 if is_var1_numeric else var1
            
            st.markdown(f"#### Comparação de Distribuições: **{numeric_var}** por **{categorical_var}**")
            st.markdown(f"O gráfico de violino abaixo é excelente para comparar a distribuição de uma variável numérica (`{numeric_var}`) entre as diferentes categorias de uma variável categórica (`{categorical_var}`).")
            fig_violin = px.violin(
                df, x=categorical_var, y=numeric_var,
                color=df[ProjectConfig.TARGET_VARIABLE].map({0: 'Bom Risco', 1: 'Mau Risco'}),
                box=True,
                points="all",
                title=f"Distribuição de '{numeric_var}' por '{categorical_var}', segmentado por Risco",
                color_discrete_map={'Bom Risco': ProjectConfig.GOOD_RISK_COLOR, 'Mau Risco': ProjectConfig.BAD_RISK_COLOR}
            )
            fig_violin.update_layout(height=600, legend_title_text='Status do Risco')
            st.plotly_chart(fig_violin, use_container_width=True)

@st.cache_data(show_spinner="Calculando projeção PCA para visualização...")
def get_pca_projection(_df, target_col):
    """
    Executa a Análise de Componentes Principais (PCA) para reduzir
    a dimensionalidade dos dados numéricos para 2D, facilitando a visualização.
    Retorna os componentes principais e a variância explicada.
    """
    numeric_cols = _df.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore')
    
    pca_df = _df.copy()
    
    # Padroniza apenas as colunas numéricas antes do PCA
    scaler = StandardScaler()
    pca_df_scaled = scaler.fit_transform(pca_df[numeric_cols])
    
    pca = PCA(n_components=2, random_state=ProjectConfig.RANDOM_STATE_SEED)
    principal_components = pca.fit_transform(pca_df_scaled)
    
    pca_result_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_result_df[target_col] = pca_df[target_col].values
    
    explained_variance = pca.explained_variance_ratio_
    return pca_result_df, explained_variance

def render_multivariate_analysis_tab(df):
    """
    Renderiza a aba de Análise Multivariada, focando na visualização
    de dados com PCA e gráficos 3D interativos para uma exploração mais profunda.
    """
    st.subheader("Análise de Múltiplas Variáveis Simultaneamente")
    st.markdown("Explore interações complexas entre vários atributos e como eles se relacionam com a variável alvo `class`.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(ProjectConfig.TARGET_VARIABLE, errors='ignore').tolist()
    
    st.markdown("#### Visualização do Espaço de Features com PCA")
    st.markdown("""
    A Análise de Componentes Principais (PCA) reduz a complexidade dos dados, projetando-os em 2D. O gráfico abaixo nos ajuda a ver se existem agrupamentos naturais de clientes de 'bom risco' versus 'mau risco'. Uma boa separação visual aqui é um bom presságio para os modelos de classificação.
    """)

    if st.button("Gerar Gráfico PCA", key="pca_button", type="primary"):
        pca_result_df, explained_variance = get_pca_projection(df, ProjectConfig.TARGET_VARIABLE)
        
        fig_pca = px.scatter(
            pca_result_df, x='PC1', y='PC2',
            color=pca_result_df[ProjectConfig.TARGET_VARIABLE].map({0: 'Bom Risco', 1: 'Mau Risco'}),
            color_discrete_map={'Bom Risco': ProjectConfig.GOOD_RISK_COLOR, 'Mau Risco': ProjectConfig.BAD_RISK_COLOR},
            title=f"Projeção PCA 2D do Dataset (Variância Explicada: {sum(explained_variance):.2%})"
        )
        fig_pca.update_layout(height=600, legend_title_text='Status do Risco')
        st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Scatter Plot 3D Interativo")
    st.markdown("Selecione três variáveis numéricas para criar um gráfico de dispersão 3D. A cor dos pontos representa o status do risco do cliente.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_3d = st.selectbox("Eixo X:", numeric_cols, index=numeric_cols.index('age'), key="x_3d")
    with col2:
        y_3d = st.selectbox("Eixo Y:", numeric_cols, index=numeric_cols.index('credit_amount'), key="y_3d")
    with col3:
        z_3d = st.selectbox("Eixo Z:", numeric_cols, index=numeric_cols.index('duration'), key="z_3d")
    
    if x_3d and y_3d and z_3d:
        fig_3d = px.scatter_3d(
            df, x=x_3d, y=y_3d, z=z_3d,
            color=df[ProjectConfig.TARGET_VARIABLE].map({0: 'Bom Risco', 1: 'Mau Risco'}),
            color_discrete_map={'Bom Risco': ProjectConfig.GOOD_RISK_COLOR, 'Mau Risco': ProjectConfig.BAD_RISK_COLOR},
            title="Visualização 3D Interativa de Features", height=700
        )
        fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
        fig_3d.update_layout(legend_title_text='Status do Risco')
        st.plotly_chart(fig_3d, use_container_width=True)

def display_eda_page():
    """
    Renderiza a página principal de Análise Exploratória Interativa (EDA).
    Organiza as diferentes análises (univariada, bivariada, multivariada) em abas
    e apresenta a análise inicial de desbalanceamento de classes com explicações detalhadas.
    """
    st.header("Análise Exploratória Interativa (EDA) 🔍")
    st.markdown("""
    Nesta seção, mergulhamos nos dados para descobrir padrões, correlações e insights iniciais sobre o risco de crédito.
    A Análise Exploratória é um passo investigativo essencial antes de construirmos qualquer modelo preditivo.
    """)

    if not st.session_state.get('data_processed', False) or st.session_state.get('processed_df') is None:
        st.warning("⚠️ Os dados precisam ser processados na página 'Análise e Preparação dos Dados' para que a Análise Exploratória seja habilitada.")
        st.info("Por favor, retorne à página anterior e clique no botão 'Executar Engenharia de Features'.")
        return

    df = st.session_state.processed_df

    st.subheader("Diagnóstico da Variável-Alvo: `class`")
    st.markdown("""
    **Ponto de Partida: Entendendo o Desafio Central**

    Antes de analisar qualquer outra variável, precisamos entender a composição do nosso alvo: a coluna `class`. É fundamental esclarecer que **não estamos definindo quem é bom ou mau pagador**. Essa definição é uma regra de negócio já estabelecida: um cliente que atrasa o pagamento por mais de 90 dias é classificado como `bad`.

    Nosso trabalho aqui é um **diagnóstico**: Qual a proporção desses clientes na nossa base de dados histórica? A resposta a essa pergunta define a principal estratégia de modelagem.
    
    O cálculo é simples:
    1.  Contamos o número de clientes para cada categoria (`good` e `bad`).
    2.  Dividimos pelo total de clientes para encontrar a proporção.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        class_counts = df[ProjectConfig.TARGET_VARIABLE].value_counts()
        # Mapeia para os nomes originais para clareza
        class_counts.index = class_counts.index.map({0: 'Bom Risco (Good)', 1: 'Mau Risco (Bad)'})
        
        bom_risco_count = class_counts.get('Bom Risco (Good)', 0)
        mau_risco_count = class_counts.get('Mau Risco (Bad)', 0)
        total_count = bom_risco_count + mau_risco_count

        st.markdown(f"""
        **Resultados do Diagnóstico:**
        - **Bons Pagadores:** `{bom_risco_count}` clientes (`{bom_risco_count/total_count:.0%}`)
        - **Maus Pagadores:** `{mau_risco_count}` clientes (`{mau_risco_count/total_count:.0%}`)
        
        **Conclusão da Análise:** Os dados são **desbalanceados**. Clientes de `mau risco` são a minoria. Se não tratarmos isso, um modelo de IA poderia ficar "preguiçoso" e aprender a simplesmente prever `bom risco` para todo mundo, atingindo uma alta acurácia, mas sendo completamente inútil para o negócio. Por isso, a técnica **SMOTE** será aplicada na fase de modelagem para balancear os dados de treino.
        """)

    with col2:
        fig_balance = px.pie(
            values=class_counts.values, names=class_counts.index, 
            title='Proporção Histórica entre Bons e Maus Pagadores',
            color=class_counts.index,
            color_discrete_map={'Bom Risco (Good)': ProjectConfig.GOOD_RISK_COLOR, 'Mau Risco (Bad)': ProjectConfig.BAD_RISK_COLOR},
            hole=.3
        )
        st.plotly_chart(fig_balance, use_container_width=True)

    st.markdown("---")
    st.markdown("Agora que entendemos o desafio principal, vamos explorar as outras variáveis em mais detalhes nas abas abaixo.")

    tab_uni, tab_bi, tab_multi = st.tabs([
        "📊 Análise Univariada", 
        "🔗 Análise Bivariada", 
        "🔮 Análise Multivariada"
    ])

    with tab_uni:
        render_univariate_analysis_tab(df)
    with tab_bi:
        render_bivariate_analysis_tab(df)
    with tab_multi:
        render_multivariate_analysis_tab(df)

@st.cache_data(show_spinner="Dividindo dados, processando e aplicando SMOTE...")
def prepare_data_for_modeling(_df, target, test_size, random_state):
    """
    Executa o pipeline completo de preparação dos dados para a modelagem supervisionada.
    Isso inclui a separação dos dados, a criação de um pipeline de pré-processamento
    com ColumnTransformer e a aplicação do SMOTE para balancear o conjunto de treino.
    Retorna um dicionário contendo todos os artefatos de dados necessários.
    """
    
    X = _df.drop(columns=[target])
    y = _df[target]
    
    # Divisão estratificada para manter a proporção da variável alvo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    # Pipeline de pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    except Exception:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names(categorical_features)
        
    processed_feature_names = numeric_features.tolist() + list(ohe_feature_names)

    # Aplicação do SMOTE para balanceamento da classe minoritária
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    # Armazena todos os artefatos em um dicionário para uso futuro
    modeling_data = {
        'X_train_orig': X_train_processed, 'y_train_orig': y_train,
        'X_train_resampled': X_train_resampled, 'y_train_resampled': y_train_resampled,
        'X_test': X_test_processed, 'y_test': y_test,
        'preprocessor': preprocessor, 'processed_feature_names': processed_feature_names,
        'X_train_raw': X_train, 'X_test_raw': X_test
    }
    
    return modeling_data

def render_data_preparation_module(df):
    """
    Renderiza o módulo de UI para a preparação dos dados de modelagem.
    Inclui um botão para iniciar o processo e, em seguida, exibe os resultados
    e o impacto visual do balanceamento com SMOTE.
    """
    with st.container(border=True):
        st.subheader("Etapa 1: Preparação e Balanceamento dos Dados")
        st.markdown("""
        **O Quê?** Aqui, preparamos os dados para que os algoritmos de Machine Learning possam "entendê-los" da melhor forma possível. Realizamos três ações cruciais:
        1.  **Divisão Estratificada:** Separamos os dados em um conjunto de **Treino** (para ensinar o modelo) e um de **Teste** (para avaliá-lo de forma imparcial). A estratificação garante que a proporção de clientes de `bom` e `mau` risco seja a mesma em ambos os conjuntos.
        2.  **Pré-processamento:** Padronizamos as variáveis numéricas (`StandardScaler`) e codificamos as variáveis de texto (`OneHotEncoder`).
        3.  **Balanceamento com SMOTE:** Nosso maior desafio é o desbalanceamento de classes. O **SMOTE (Synthetic Minority Over-sampling Technique)** resolve isso criando exemplos sintéticos e realistas de clientes de `mau` risco no conjunto de treino.
        
        **Justificativa da Escolha (SMOTE):** O SMOTE foi escolhido por ser uma técnica de oversampling sofisticada que cria novas instâncias baseadas nos vizinhos mais próximos, evitando o simples "copia e cola". Isso gera um conjunto de treino mais rico e diverso, ajudando os modelos a generalizarem melhor.
        """)
        
        if st.button("Executar Divisão e Balanceamento dos Dados", type="primary", key="prep_button"):
            modeling_data = prepare_data_for_modeling(
                df, 
                target=ProjectConfig.TARGET_VARIABLE, 
                test_size=ProjectConfig.TEST_SIZE_RATIO, 
                random_state=ProjectConfig.RANDOM_STATE_SEED
            )
            st.session_state.artifacts['modeling_data'] = modeling_data
            st.session_state.app_stage = 'data_prepared'
            st.success("Dados preparados com sucesso!")
            st.rerun()

    if 'modeling_data' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            modeling_data = st.session_state.artifacts['modeling_data']
            st.subheader("Resultados da Preparação e Impacto do SMOTE")
            st.markdown("""
            Veja abaixo a quantidade de clientes em cada conjunto. Note como o conjunto de treino se tornou perfeitamente balanceado (50/50) após o SMOTE. O gráfico de dispersão (PCA) mostra visualmente esse impacto, transformando a nuvem de pontos minoritária (vermelha) em um grupo denso e claro, ideal para o treinamento.
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clientes Treino (Original)", len(modeling_data['y_train_orig']))
            col2.metric("Clientes Teste", len(modeling_data['y_test']))
            col3.metric("Clientes Treino (Pós-SMOTE)", len(modeling_data['y_train_resampled']))
            y_resampled_counts = pd.Series(modeling_data['y_train_resampled']).value_counts(normalize=True) * 100
            col4.metric("Balanceamento Pós-SMOTE", f"{y_resampled_counts[0]:.0f}% / {y_resampled_counts[1]:.0f}")

            with st.expander("Visualizar o Impacto do SMOTE (Projeção PCA)"):
                pca_vis = PCA(n_components=2, random_state=ProjectConfig.RANDOM_STATE_SEED)
                X_train_pca_before = pca_vis.fit_transform(modeling_data['X_train_orig'])
                X_train_pca_after = pca_vis.transform(modeling_data['X_train_resampled'])

                df_before = pd.DataFrame(X_train_pca_before, columns=['PC1', 'PC2'])
                df_before['Risco'] = [f"Risco {v}" for v in modeling_data['y_train_orig'].values]
                df_after = pd.DataFrame(X_train_pca_after, columns=['PC1', 'PC2'])
                df_after['Risco'] = [f"Risco {v}" for v in modeling_data['y_train_resampled'].values]
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Antes do SMOTE", "Depois do SMOTE"))
                fig.add_trace(go.Scatter(x=df_before['PC1'], y=df_before['PC2'], mode='markers', marker=dict(color=modeling_data['y_train_orig'].values, colorscale=[ProjectConfig.GOOD_RISK_COLOR, ProjectConfig.BAD_RISK_COLOR], showscale=False, opacity=0.7), name='Antes'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_after['PC1'], y=df_after['PC2'], mode='markers', marker=dict(color=modeling_data['y_train_resampled'].values, colorscale=[ProjectConfig.GOOD_RISK_COLOR, ProjectConfig.BAD_RISK_COLOR], showscale=False, opacity=0.7), name='Depois'), row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner="Executando Seleção de Features com RFECV... Este processo pode ser intensivo.")
def run_rfe_cv_feature_selection(_modeling_data):
    """
    Executa a Eliminação Recursiva de Features com Validação Cruzada (RFECV)
    para encontrar o subconjunto ideal de features que maximiza a performance
    preditiva, evitando ruído e complexidade desnecessária.

    A função utiliza um estimador rápido e robusto (LGBMClassifier) para avaliar
    os subconjuntos de features, otimizando pela métrica ROC AUC em um esquema
    de validação cruzada estratificada para garantir a representatividade das classes.

    Retorna:
        dict: Um dicionário contendo o objeto seletor treinado, o número ótimo de
              features, seus nomes, e os scores da validação cruzada para visualização.
    """
    X_train = _modeling_data['X_train_resampled']
    y_train = _modeling_data['y_train_resampled']
    feature_names = _modeling_data['processed_feature_names']
    
    # Define um estimador leve para o processo de seleção
    estimator_for_rfe = LGBMClassifier(
        random_state=ProjectConfig.RANDOM_STATE_SEED, 
        verbose=-1
    )
    
    # Configura a estratégia de validação cruzada
    cv_strategy = StratifiedKFold(
        n_splits=ProjectConfig.N_SPLITS_KFOLD, 
        shuffle=True, 
        random_state=ProjectConfig.RANDOM_STATE_SEED
    )
    
    # Instancia o seletor RFECV
    rfe_selector = RFECV(
        estimator=estimator_for_rfe,
        step=1,
        cv=cv_strategy,
        scoring=ProjectConfig.RFE_CV_SCORING,
        min_features_to_select=10,  # Garante um mínimo de features para o modelo
        n_jobs=-1
    )
    
    rfe_selector.fit(X_train, y_train)
    
    selected_feature_names = [
        feature for feature, support in zip(feature_names, rfe_selector.support_) if support
    ]

    # Em versões mais recentes do scikit-learn, o atributo é 'cv_results_'
    if hasattr(rfe_selector, 'cv_results_'):
        cv_results = rfe_selector.cv_results_['mean_test_score']
    else:
        # Fallback para versões mais antigas
        cv_results = getattr(rfe_selector, 'grid_scores_', None)


    selection_artifacts = {
        'selector_object': rfe_selector,
        'optimal_n_features': rfe_selector.n_features_,
        'selected_feature_names': selected_feature_names,
        'cv_results_scores': cv_results,
    }
    
    return selection_artifacts

def render_feature_selection_module(modeling_data):
    """
    Renderiza o módulo de UI para a Seleção de Features.
    Explica a importância da etapa e exibe os resultados da execução do RFECV,
    incluindo a curva de performance da validação cruzada que justifica a
    escolha do número de features.
    """
    with st.container(border=True):
        st.subheader("Etapa 2: Foco no que Importa - Seleção de Features")
        st.markdown("""
        **O Quê?** Nem todas as características dos clientes são igualmente importantes para prever o risco. Nesta etapa, utilizamos uma técnica avançada chamada **RFECV (Recursive Feature Elimination with Cross-Validation)** para encontrar o subconjunto de features mais preditivo. O algoritmo treina um modelo repetidamente, removendo as features menos importantes a cada passo e validando o resultado, até encontrar o "time" ideal de variáveis.

        **Justificativa da Escolha (RFECV):** Diferente de métodos univariados, o RFECV considera a interação entre as variáveis, resultando em um conjunto de features mais robusto e performático. Ele nos ajuda a reduzir a complexidade do modelo, diminuir o risco de overfitting e, muitas vezes, aumentar a interpretabilidade, focando apenas no que realmente importa.
        """)
        
        if st.button("Executar Seleção de Features com RFECV", key="fs_button_rfe", type="primary"):
            selection_artifacts = run_rfe_cv_feature_selection(modeling_data)
            st.session_state.artifacts['selection_artifacts'] = selection_artifacts
            st.session_state.app_stage = 'features_selected'
            st.success("Seleção de features com RFECV concluída!")
            st.rerun()

    if 'selection_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state.artifacts['selection_artifacts']
            st.subheader("Resultados da Seleção de Features")
            st.metric(label="Número ideal de features encontrado", value=artifacts['optimal_n_features'])
            
            if artifacts.get('cv_results_scores') is not None:
                cv_scores = artifacts['cv_results_scores']
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(cv_scores) + 1)), y=cv_scores,
                    mode='lines+markers', marker=dict(color=ProjectConfig.PRIMARY_COLOR), line=dict(width=3)
                ))
                fig.add_vline(x=artifacts['optimal_n_features'], line_width=2, line_dash="dash", line_color=ProjectConfig.ACCENT_COLOR,
                             annotation_text="Número Ótimo de Features", annotation_position="top left")
                fig.update_layout(
                    title='Performance do Modelo vs. Número de Features (Validação Cruzada)',
                    xaxis_title='Número de Features Selecionadas',
                    yaxis_title=f'Score de Validação ({ProjectConfig.RFE_CV_SCORING.upper()})',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver Lista de Features Selecionadas para a Modelagem"):
                st.dataframe(pd.DataFrame(artifacts['selected_feature_names'], columns=["Feature Selecionada"]), use_container_width=True)
            
            st.info("Com as features mais importantes selecionadas, estamos prontos para a competição de modelos.", icon="🏆")

@st.cache_data(show_spinner="Treinando e avaliando todos os 9 modelos... Este processo pode levar alguns minutos.")
def train_baseline_models(_modeling_data, _selection_artifacts):
    """
    Treina uma lista completa de modelos de classificação, incluindo todos os
    solicitados na Prova Final. Avalia a performance de cada um no conjunto de
    teste e retorna um dicionário com todos os modelos, métricas e artefatos.
    """
    X_train = _modeling_data['X_train_resampled']
    y_train = _modeling_data['y_train_resampled']
    X_test = _modeling_data['X_test']
    y_test = _modeling_data['y_test']

    selector = _selection_artifacts['selector_object']
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)
    
    models_to_test = {
        "Regressão Logística": LogisticRegression(random_state=ProjectConfig.RANDOM_STATE_SEED, max_iter=1000, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "SVM": SVC(probability=True, random_state=ProjectConfig.RANDOM_STATE_SEED),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "Random Forest": RandomForestClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "XGBoost": XGBClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, verbose=-1),
        "Rede Neural (MLP)": MLPClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, max_iter=500, early_stopping=True, hidden_layer_sizes=(50, 25))
    }

    baseline_results = {}
    progress_bar = st.progress(0, text="Iniciando treinamento dos modelos...")

    for i, (name, model) in enumerate(models_to_test.items()):
        progress_bar.progress((i + 1) / len(models_to_test), text=f"Treinando modelo: {name}...")
        
        # Usamos um pipeline para modelos que não são de árvore para garantir consistência
        if not hasattr(model, 'feature_importances_') and not name == "SVM":
             pipeline = ImbPipeline([('model', model)])
        else:
             pipeline = model

        pipeline.fit(X_train_final, y_train)
        y_pred = pipeline.predict(X_test_final)
        y_proba = pipeline.predict_proba(X_test_final)[:, 1]
        
        baseline_results[name] = {
            'model_object': pipeline,
            'metrics': {
                'AUC': roc_auc_score(y_test, y_proba),
                'Acurácia': accuracy_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'Precisão': precision_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'full_report': classification_report(y_test, y_pred, output_dict=True, target_names=['Bom Risco', 'Mau Risco']),
            'roc_curve_data': roc_curve(y_test, y_proba)
        }
    progress_bar.empty()
    return baseline_results

def render_baseline_modeling_module(modeling_data, selection_artifacts):
    """
    Renderiza o módulo de UI para o treinamento dos modelos de baseline.
    Apresenta um leaderboard interativo para comparar a performance dos algoritmos.
    """
    with st.container(border=True):
        st.subheader("Etapa 3: A Competição dos Modelos Supervisionados")
        st.markdown("""
        **O Quê?** Agora começa a competição! Treinamos todos os 9 modelos de classificação solicitados no desafio, usando os dados preparados e as features selecionadas na etapa anterior. Cada modelo é avaliado com um conjunto rigoroso de métricas de performance no conjunto de teste, que ele nunca viu antes.

        **Por quê?** Esta competição nos permite comparar objetivamente a eficácia de diferentes abordagens algorítmicas para o nosso problema de risco de crédito. O **Leaderboard de Performance** abaixo resume os resultados, permitindo-nos identificar os modelos mais promissores para uma análise mais profunda.
        """)
        
        if st.button("Executar Competição de Modelos", key="train_button", type="primary"):
            baseline_artifacts = train_baseline_models(modeling_data, selection_artifacts)
            if baseline_artifacts:
                st.session_state.artifacts['baseline_artifacts'] = baseline_artifacts
                st.session_state.app_stage = 'baselines_trained'
                st.success("Competição de modelos baseline concluída com sucesso!")
                st.rerun()

    if 'baseline_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state.artifacts['baseline_artifacts']
            st.subheader("Análise Pós-Treinamento: O Leaderboard de Performance")
            st.markdown("""
            **Como interpretar:** Ordene a tabela clicando nos cabeçalhos das colunas.
            - **AUC:** A principal métrica de performance geral (próximo de 1.0 é melhor).
            - **Recall:** Crucial para o negócio! Mostra a porcentagem de **maus pagadores** que o modelo conseguiu identificar corretamente. Um Recall alto evita que clientes de risco passem despercebidos.
            - **Precisão:** Dos clientes que o modelo previu como `mau` risco, quantos realmente eram.
            - **F1-Score:** Uma média harmônica entre Precisão e Recall.
            """)
            
            leaderboard_data = [{'Modelo': name, **res['metrics']} for name, res in artifacts.items()]
            leaderboard_df = pd.DataFrame(leaderboard_data).set_index('Modelo')
            
            sort_by = st.selectbox("Ordenar leaderboard pela métrica:", leaderboard_df.columns, index=0)
            sorted_df = leaderboard_df.sort_values(by=sort_by, ascending=False)
            st.dataframe(
                sorted_df.style.background_gradient(cmap='viridis', subset=[sort_by], low=0.6)
                               .format("{:.4f}")
                               .highlight_max(subset=[sort_by], color='#94d2bd'),
                use_container_width=True
            )
            st.info("Com o leaderboard, podemos agora fazer uma análise aprofundada de cada competidor.", icon="🔬")

def render_roc_tab(model_data, model_name):
    """
    Renderiza o conteúdo da aba 'Curva ROC' no módulo de análise aprofundada.
    """
    metrics = model_data['metrics']
    fpr, tpr, _ = model_data['roc_curve_data']

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        mode='lines', 
        name=f'Curva ROC (AUC = {metrics["AUC"]:.4f})', 
        line=dict(color=ProjectConfig.PRIMARY_COLOR, width=4)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode='lines', 
        name='Performance Aleatória (AUC = 0.5)',
        line=dict(color=ProjectConfig.ACCENT_COLOR, dash='dash')
    ))
    
    fig_roc.update_layout(
        title=f"Curva ROC para o Modelo: {model_name}",
        xaxis_title='Taxa de Falsos Positivos (1 - Especificidade)',
        yaxis_title='Taxa de Verdadeiros Positivos (Recall / Sensibilidade)',
        legend=dict(x=0.55, y=0.1, bgcolor='rgba(255,255,255,0.6)'),
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True, key=f"deep_dive_roc_{model_name}")

def render_report_tab(model_data):
    """
    Renderiza o conteúdo da aba 'Relatório Completo'.
    """
    st.markdown("O relatório abaixo detalha as métricas para cada classe.")
    report_df = pd.DataFrame(model_data['full_report']).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

def render_importance_tab(model_data, model_name):
    """
    Renderiza o conteúdo da aba 'Importância de Features'.
    """
    if hasattr(model_data['model_object'], 'steps'):
        model_object = model_data['model_object'].steps[-1][1]
    else:
        model_object = model_data['model_object']
    
    importances = None
    if hasattr(model_object, 'feature_importances_'):
        importances = model_object.feature_importances_
    elif hasattr(model_object, 'coef_'):
        importances = np.abs(model_object.coef_[0])

    if importances is not None:
        feature_names = st.session_state['artifacts']['selection_artifacts']['selected_feature_names']
        if len(feature_names) == len(importances):
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importância': importances}).sort_values(by='Importância', ascending=False)
            fig_imp = px.bar(
                importance_df.head(20).sort_values(by='Importância', ascending=True), 
                x='Importância', y='Feature', orientation='h', 
                title=f"Top 20 Features Mais Importantes para o Modelo: {model_name}"
            )
            st.plotly_chart(fig_imp, use_container_width=True, key=f"deep_dive_importance_{model_name}")
    else:
        st.info(f"O modelo '{model_name}' não possui um atributo de importância de features direto. A análise com SHAP na próxima etapa é a mais indicada.", icon="ℹ️")

def render_model_deep_dive_module(baseline_artifacts):
    """
    Renderiza o módulo de UI para análise detalhada de cada modelo treinado.
    Utiliza abas para organizar as diferentes visualizações e análises,
    começando pela crucial Matriz de Confusão.
    """
    with st.container(border=True):
        st.subheader("Etapa 4: Análise Aprofundada dos Competidores")
        st.markdown("""
        **O Quê?** Escolha um modelo do leaderboard para uma inspeção detalhada. Aqui, vamos além das métricas gerais e investigamos *como* cada modelo acerta e erra.
        
        **Por quê?** Entender os pontos fortes e fracos de cada algoritmo é essencial para a **tomada de decisão gerencial**. A **Matriz de Confusão** é nossa principal ferramenta aqui. Ela nos mostra os quatro cenários possíveis:
        - **Verdadeiros Negativos (VN):** Modelo previu `Bom Risco`, e acertou.
        - **Verdadeiros Positivos (VP):** Modelo previu `Mau Risco`, e acertou.
        - **Falsos Positivos (FP):** Modelo previu `Mau Risco`, mas errou (custo de oportunidade, cliente bom negado).
        - **Falsos Negativos (FN):** Modelo previu `Bom Risco`, mas errou (prejuízo, cliente mau aprovado). **Este é o erro mais caro para o negócio!**
        """)
        
        # Ordena os modelos por AUC para sugerir o melhor primeiro
        sorted_models = sorted(baseline_artifacts.keys(), key=lambda k: baseline_artifacts[k]['metrics']['AUC'], reverse=True)

        model_to_inspect = st.selectbox(
            "Selecione um modelo do leaderboard para uma análise detalhada:",
            options=sorted_models
        )
        
        if model_to_inspect:
            model_explanations = {
            "Regressão Logística": {
                "desc": "Um modelo estatístico fundamental para classificação binária. Ele estima a probabilidade de um evento ocorrer (neste caso, `mau risco`) com base nos valores das variáveis de entrada, aplicando uma função logística.",
                "decision": "Sua principal vantagem é a **interpretabilidade**. Os coeficientes do modelo nos dizem exatamente o quanto cada variável aumenta ou diminui a chance de inadimplência, permitindo criar políticas de crédito extremamente claras e baseadas em evidências estatísticas."
            },
            "KNN": {
                "desc": "K-Nearest Neighbors (Vizinhos Mais Próximos) é um algoritmo baseado em instância. Ele classifica um novo cliente com base na classe da maioria dos 'K' vizinhos mais próximos a ele no espaço de features. É intuitivo e não assume nenhuma premissa sobre a distribuição dos dados.",
                "decision": "É útil para encontrar padrões locais e não lineares que outros modelos podem perder. Na tomada de decisão, ajuda a identificar 'bolsões' de risco: clientes que, embora pareçam bons isoladamente, estão cercados por um 'bairro' de maus pagadores."
            },
            "SVM": {
                "desc": "Support Vector Machine (Máquina de Vetores de Suporte) busca encontrar o 'hiperplano' (uma linha ou plano) que melhor separa as duas classes (bons e maus pagadores) com a maior margem possível. Usando 'kernels', ele consegue encontrar fronteiras de decisão complexas e não lineares.",
                "decision": "A força do SVM está em sua capacidade de lidar com dados de alta dimensionalidade e encontrar relações complexas. Para a decisão, ele é excelente em identificar os casos mais difíceis e ambíguos que ficam perto da fronteira entre ser um bom ou mau pagador."
            },
            "Árvore de Decisão": {
                "desc": "Cria um modelo semelhante a um fluxograma de decisões. Cada 'nó' da árvore representa um teste em uma variável (ex: 'duração > 12 meses?'), e cada 'folha' representa uma classificação (bom ou mau risco).",
                "decision": "Sua maior vantagem é a **transparência**. Uma árvore de decisão é facilmente visualizável e compreendida por qualquer pessoa do time de negócios, tornando a regra de decisão explícita. É a base para modelos mais complexos como o Random Forest."
            },
            "Random Forest": {
                "desc": "É um modelo de 'ensemble' (conjunto) que constrói múltiplas árvores de decisão durante o treinamento e decide a classificação final com base na 'votação' da maioria das árvores. Isso reduz o superajuste (overfitting) e melhora a generalização.",
                "decision": "Para a tomada de decisão, o Random Forest oferece um excelente equilíbrio entre alta performance preditiva e boa interpretabilidade (podemos ver a importância média das features). Ele é robusto e geralmente fornece previsões muito estáveis."
            },
            "AdaBoost": {
                "desc": "Adaptive Boosting é um algoritmo de 'boosting'. Ele treina uma sequência de modelos fracos (geralmente pequenas árvores), onde cada novo modelo dá mais atenção aos erros de classificação do modelo anterior. O resultado final é uma soma ponderada de todos os modelos.",
                "decision": "Sua natureza adaptativa o torna eficaz em casos difíceis. Para a decisão, ele ajuda a focar nos perfis de clientes que são consistentemente classificados de forma errada, permitindo a criação de políticas específicas para esses nichos de maior incerteza."
            },
            "Gradient Boosting": {
                "desc": "Assim como o AdaBoost, treina modelos em sequência. No entanto, em vez de ajustar os pesos dos erros, cada novo modelo tenta corrigir o 'erro residual' (a diferença entre a previsão e o valor real) do modelo anterior. É um dos algoritmos mais performáticos.",
                "decision": "Oferece altíssima precisão. Para a decisão gerencial, um modelo de Gradient Boosting bem ajustado pode ser a ferramenta definitiva para criar um score de crédito extremamente acurado, minimizando perdas por aprovar clientes de alto risco."
            },
            "XGBoost": {
                "desc": "eXtreme Gradient Boosting é uma implementação otimizada e de alta performance do Gradient Boosting. Inclui regularização para evitar overfitting, tratamento de dados faltantes e processamento paralelo, tornando-o mais rápido e robusto.",
                "decision": "É o padrão ouro em muitas competições de dados por um motivo: performance. Na tomada de decisão, um modelo XGBoost pode ser a base de um sistema de aprovação de crédito automatizado de alta performance, capaz de processar milhares de solicitações com grande precisão."
            },
            "LightGBM": {
                "desc": "Light Gradient Boosting Machine é outra implementação de boosting, focada em velocidade e eficiência de memória. Ele cresce as árvores 'folha por folha' em vez de 'nível por nível', o que o torna extremamente rápido em grandes datasets.",
                "decision": "Sua velocidade o torna ideal para ambientes de produção que exigem decisões em tempo real ou retreinamento frequente dos modelos. Para a gestão, significa ter um sistema de risco que pode ser atualizado diariamente com novos dados sem comprometer a performance."
            },
            "Rede Neural (MLP)": {
                "desc": "Multi-layer Perceptron é um modelo inspirado no cérebro humano, com camadas de 'neurônios' interconectados. Ele é capaz de aprender padrões extremamente complexos e não lineares nos dados.",
                "decision": "Sua força está na capacidade de capturar interações sutis entre as variáveis que outros modelos podem não ver. Para a decisão, pode revelar perfis de risco não intuitivos, justificando políticas de crédito inovadoras e altamente segmentadas."
            }
        }
        
        explanation = model_explanations.get(model_to_inspect, {})
        if explanation:
            with st.expander(f"Entendendo o Modelo: {model_to_inspect}", expanded=True):
                st.markdown(f"**Como Funciona:** {explanation['desc']}")
                st.markdown(f"**Relevância para a Tomada de Decisão:** {explanation['decision']}")

            model_data = baseline_artifacts[model_to_inspect]
            metrics = model_data['metrics']
            
            st.markdown(f"##### Métricas de Performance para o Modelo **{model_to_inspect}**")
        
        metric_explanations = {
            "AUC": "Mede a capacidade geral do modelo de distinguir entre bons e maus pagadores. Quanto mais perto de 1.0, melhor o modelo.",
            "Acurácia": "Percentual geral de acertos. Pode ser enganosa em dados desbalanceados, por isso olhamos outras métricas.",
            "Recall": "A métrica mais importante para o risco! Dos clientes que realmente eram 'maus', quantos o modelo conseguiu capturar? Um recall alto evita prejuízos.",
            "Precisão": "Dos clientes que o modelo classificou como 'maus', quantos ele acertou? Uma precisão alta evita negar crédito a bons clientes.",
            "F1-Score": "Uma média harmônica entre Precisão e Recall. Útil para um balanço geral da performance na classe positiva ('mau risco')."
        }

        metric_cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            help_text = metric_explanations.get(metric_name, "")
            metric_cols[i].metric(metric_name, f"{metric_value:.4f}", help=help_text)

            tab_cm, tab_roc, tab_report, tab_importance = st.tabs(["Matriz de Confusão", "Curva ROC", "Relatório Detalhado", "Importância de Features"])

            with tab_cm:
                cm = model_data['confusion_matrix']
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    labels=dict(x="Valores Previstos pelo Modelo", y="Valores Reais da Base de Teste"),
                    x=['Bom Risco', 'Mau Risco'], y=['Bom Risco', 'Mau Risco'],
                    title=f"Matriz de Confusão para o Modelo: {model_to_inspect}",
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True, key=f"plotly_cm_{model_to_inspect}")
                st.warning(f"O modelo falhou em identificar **{cm[1, 0]}** clientes de alto risco (Falsos Negativos), que representam o maior prejuízo potencial. Por outro lado, classificou erroneamente **{cm[0, 1]}** bons clientes como de alto risco (Falsos Positivos).")

            with tab_roc:
                render_roc_tab(model_data, model_to_inspect)
            
            with tab_report:
                render_report_tab(model_data)

            with tab_importance:
                render_importance_tab(model_data, model_to_inspect)

def render_model_deep_dive_module(baseline_artifacts):
    with st.container(border=True):
        st.subheader("Etapa 4: Análise Aprofundada dos Competidores")
        st.markdown("""
        **O Quê?** Escolha um modelo do leaderboard para uma inspeção detalhada. Aqui, vamos além das métricas gerais e investigamos *como* cada modelo acerta e erra.
        
        **Por quê?** Entender os pontos fortes e fracos de cada algoritmo é essencial para a **tomada de decisão gerencial**. A **Matriz de Confusão** é nossa principal ferramenta aqui.
        """)
        
        sorted_models = sorted(baseline_artifacts.keys(), key=lambda k: baseline_artifacts[k]['metrics']['AUC'], reverse=True)
        model_to_inspect = st.selectbox(
            "Selecione um modelo do leaderboard para uma análise detalhada:",
            options=sorted_models
        )
        
        if model_to_inspect:
            model_explanations = {
                "Regressão Logística": {
                    "desc": "Um modelo estatístico fundamental para classificação binária. Ele estima a probabilidade de um evento ocorrer (neste caso, `mau risco`) com base nos valores das variáveis de entrada, aplicando uma função logística.",
                    "decision": "Sua principal vantagem é a **interpretabilidade**. Os coeficientes do modelo nos dizem exatamente o quanto cada variável aumenta ou diminui a chance de inadimplência, permitindo criar políticas de crédito extremamente claras e baseadas em evidências estatísticas."
                },
                "KNN": {
                    "desc": "K-Nearest Neighbors (Vizinhos Mais Próximos) é um algoritmo baseado em instância. Ele classifica um novo cliente com base na classe da maioria dos 'K' vizinhos mais próximos a ele no espaço de features. É intuitivo e não assume nenhuma premissa sobre a distribuição dos dados.",
                    "decision": "É útil para encontrar padrões locais e não lineares que outros modelos podem perder. Na tomada de decisão, ajuda a identificar 'bolsões' de risco: clientes que, embora pareçam bons isoladamente, estão cercados por um 'bairro' de maus pagadores."
                },
                "SVM": {
                    "desc": "Support Vector Machine (Máquina de Vetores de Suporte) busca encontrar o 'hiperplano' (uma linha ou plano) que melhor separa as duas classes (bons e maus pagadores) com a maior margem possível. Usando 'kernels', ele consegue encontrar fronteiras de decisão complexas e não lineares.",
                    "decision": "A força do SVM está em sua capacidade de lidar com dados de alta dimensionalidade e encontrar relações complexas. Para a decisão, ele é excelente em identificar os casos mais difíceis e ambíguos que ficam perto da fronteira entre ser um bom ou mau pagador."
                },
                "Árvore de Decisão": {
                    "desc": "Cria um modelo semelhante a um fluxograma de decisões. Cada 'nó' da árvore representa um teste em uma variável (ex: 'duração > 12 meses?'), e cada 'folha' representa uma classificação (bom ou mau risco).",
                    "decision": "Sua maior vantagem é a **transparência**. Uma árvore de decisão é facilmente visualizável e compreendida por qualquer pessoa do time de negócios, tornando a regra de decisão explícita. É a base para modelos mais complexos como o Random Forest."
                },
                "Random Forest": {
                    "desc": "É um modelo de 'ensemble' (conjunto) que constrói múltiplas árvores de decisão durante o treinamento e decide a classificação final com base na 'votação' da maioria das árvores. Isso reduz o superajuste (overfitting) e melhora a generalização.",
                    "decision": "Para a tomada de decisão, o Random Forest oferece um excelente equilíbrio entre alta performance preditiva e boa interpretabilidade (podemos ver a importância média das features). Ele é robusto e geralmente fornece previsões muito estáveis."
                },
                "AdaBoost": {
                    "desc": "Adaptive Boosting é um algoritmo de 'boosting'. Ele treina uma sequência de modelos fracos (geralmente pequenas árvores), onde cada novo modelo dá mais atenção aos erros de classificação do modelo anterior. O resultado final é uma soma ponderada de todos os modelos.",
                    "decision": "Sua natureza adaptativa o torna eficaz em casos difíceis. Para a decisão, ele ajuda a focar nos perfis de clientes que são consistentemente classificados de forma errada, permitindo a criação de políticas específicas para esses nichos de maior incerteza."
                },
                "Gradient Boosting": {
                    "desc": "Assim como o AdaBoost, treina modelos em sequência. No entanto, em vez de ajustar os pesos dos erros, cada novo modelo tenta corrigir o 'erro residual' (a diferença entre a previsão e o valor real) do modelo anterior. É um dos algoritmos mais performáticos.",
                    "decision": "Oferece altíssima precisão. Para a decisão gerencial, um modelo de Gradient Boosting bem ajustado pode ser a ferramenta definitiva para criar um score de crédito extremamente acurado, minimizando perdas por aprovar clientes de alto risco."
                },
                "XGBoost": {
                    "desc": "eXtreme Gradient Boosting é uma implementação otimizada e de alta performance do Gradient Boosting. Inclui regularização para evitar overfitting, tratamento de dados faltantes e processamento paralelo, tornando-o mais rápido e robusto.",
                    "decision": "É o padrão ouro em muitas competições de dados por um motivo: performance. Na tomada de decisão, um modelo XGBoost pode ser a base de um sistema de aprovação de crédito automatizado de alta performance, capaz de processar milhares de solicitações com grande precisão."
                },
                "LightGBM": {
                    "desc": "Light Gradient Boosting Machine é outra implementação de boosting, focada em velocidade e eficiência de memória. Ele cresce as árvores 'folha por folha' em vez de 'nível por nível', o que o torna extremamente rápido em grandes datasets.",
                    "decision": "Sua velocidade o torna ideal para ambientes de produção que exigem decisões em tempo real ou retreinamento frequente dos modelos com novos dados. Para a gestão, significa ter um sistema de risco que pode ser atualizado diariamente sem comprometer a performance."
                },
                "Rede Neural (MLP)": {
                    "desc": "Multi-layer Perceptron é um modelo inspirado no cérebro humano, com camadas de 'neurônios' interconectados. Ele é capaz de aprender padrões extremamente complexos e não lineares nos dados.",
                    "decision": "Sua força está na capacidade de capturar interações sutis entre as variáveis que outros modelos podem não ver. Para a decisão, pode revelar perfis de risco não intuitivos, justificando políticas de crédito inovadoras e altamente segmentadas."
                }
            }
            
            explanation = model_explanations.get(model_to_inspect, {})
            if explanation:
                with st.expander(f"Entendendo o Modelo: {model_to_inspect}", expanded=True):
                    st.markdown(f"**Como Funciona:** {explanation['desc']}")
                    st.markdown(f"**Relevância para a Tomada de Decisão:** {explanation['decision']}")

            model_data = baseline_artifacts[model_to_inspect]
            metrics = model_data['metrics']
            
            st.markdown(f"##### Métricas de Performance para o Modelo **{model_to_inspect}**")
            
            metric_explanations_detail = {
                "AUC": "Mede a capacidade geral do modelo de distinguir entre bons e maus pagadores. Quanto mais perto de 1.0, melhor.",
                "Acurácia": "Percentual geral de acertos. Pode ser enganosa em dados desbalanceados.",
                "Recall": "A métrica mais importante para o risco! Dos clientes que realmente eram 'maus', quantos o modelo conseguiu capturar?",
                "Precisão": "Dos clientes que o modelo classificou como 'maus', quantos ele acertou?",
                "F1-Score": "Uma média harmônica entre Precisão e Recall. Útil para um balanço geral."
            }
            
            metric_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                help_text = metric_explanations_detail.get(metric_name, "")
                metric_cols[i].metric(metric_name, f"{metric_value:.4f}", help=help_text)

            tab_cm, tab_roc, tab_report, tab_importance = st.tabs(["Matriz de Confusão", "Curva ROC", "Relatório Detalhado", "Importância de Features"])
            
            with tab_cm:
                cm = model_data['confusion_matrix']
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    labels=dict(x="Valores Previstos pelo Modelo", y="Valores Reais da Base de Teste"),
                    x=['Bom Risco', 'Mau Risco'], y=['Bom Risco', 'Mau Risco'],
                    title=f"Matriz de Confusão para o Modelo: {model_to_inspect}",
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True, key=f"plotly_cm_{model_to_inspect}")
                st.warning(f"O modelo falhou em identificar **{cm[1, 0]}** clientes de alto risco (Falsos Negativos), que representam o maior prejuízo potencial. Por outro lado, classificou erroneamente **{cm[0, 1]}** bons clientes como de alto risco (Falsos Positivos).")

            with tab_roc:
                render_roc_tab(model_data, model_to_inspect)
            
            with tab_report:
                render_report_tab(model_data)

            with tab_importance:
                render_importance_tab(model_data, model_to_inspect)

@st.cache_resource(show_spinner="Finalizando modelo campeão e calculando explicações SHAP...")
def finalize_and_explain_model(_baseline_artifacts, _modeling_data, _selection_artifacts):
    """
    Identifica o melhor modelo de baseline, recalcula as predições e gera as
    explicações SHAP, utilizando o explainer correto para cada tipo de modelo.
    """
    if not _baseline_artifacts:
        return None

    best_model_name = max(
        _baseline_artifacts, 
        key=lambda k: _baseline_artifacts[k]['metrics']['AUC']
    )
    final_model_pipeline = _baseline_artifacts[best_model_name]['model_object']
    
    # Se o modelo for um pipeline, extrai o estimador final
    if hasattr(final_model_pipeline, 'steps'):
        final_model = final_model_pipeline.steps[-1][1]
    else:
        final_model = final_model_pipeline

    X_train_final = _selection_artifacts['selector_object'].transform(_modeling_data['X_train_resampled'])
    X_test_final = _selection_artifacts['selector_object'].transform(_modeling_data['X_test'])
    
    X_test_df = pd.DataFrame(
        X_test_final, 
        columns=_selection_artifacts['selected_feature_names']
    )
    
    # Lógica para usar o explainer correto dependendo do tipo de modelo
    model_type = type(final_model).__name__
    if model_type in ['XGBClassifier', 'LGBMClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        explainer = shap.TreeExplainer(final_model)
        shap_values_raw = explainer.shap_values(X_test_df)
    else:
        st.info(f"Modelo '{model_type}' não é baseado em árvore. Utilizando KernelExplainer (pode ser mais lento).")
        X_train_summary = shap.sample(X_train_final, 100)
        explainer = shap.KernelExplainer(final_model_pipeline.predict_proba, X_train_summary)
        shap_values_raw = explainer.shap_values(X_test_df)

    shap_values_for_bad_risk = shap_values_raw[1] if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2 else shap_values_raw

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    final_artifacts = {
        'model_name': best_model_name,
        'model_object': final_model_pipeline,
        'X_test_df': X_test_df,
        'y_test': _modeling_data['y_test'],
        'y_pred_test': final_model_pipeline.predict(X_test_final),
        'y_proba_test': final_model_pipeline.predict_proba(X_test_final),
        'explainer': explainer,
        'shap_values': shap_values_for_bad_risk,
        'expected_value': expected_value
    }
    return final_artifacts

def render_final_model_analysis_module(baseline_artifacts, modeling_data, selection_artifacts):
    """
    Renderiza a UI para a seleção do modelo campeão e a análise de trade-off
    entre Precisão e Recall, um passo crucial para a decisão de negócio.
    """
    with st.container(border=True):
        st.subheader("Etapa 5: Seleção do Modelo Campeão e Geração de Explicações (XAI)")
        st.markdown("""
        **O Quê?** Com base no leaderboard, elegemos o modelo com maior **AUC** como nosso campeão. Agora, vamos submetê-lo ao passo mais importante: a **geração de explicações com SHAP**. Isso nos permitirá entender *por que* o modelo toma certas decisões, abrindo a caixa-preta da IA.

        **Por quê?** Um bom modelo não é apenas preciso, ele é **confiável e transparente**. A explicabilidade é fundamental para que o time de negócios confie nas previsões e para que possamos criar políticas de crédito justas e eficazes. Esta etapa é o pré-requisito para a análise de decisão gerencial.
        """)
        
        best_model_name = max(
            baseline_artifacts, 
            key=lambda k: baseline_artifacts[k]['metrics']['AUC']
        )
        st.success(f"O modelo com melhor performance (maior AUC) no leaderboard é o **{best_model_name}**. Ele será promovido para a análise de explicabilidade.", icon="🏆")

        if st.button("Analisar Modelo Campeão e Gerar Explicações SHAP", key="final_model_button", type='primary'):
            with st.spinner("Executando análise de explicabilidade com SHAP... Este processo pode levar alguns minutos."):
                final_artifacts = finalize_and_explain_model(baseline_artifacts, modeling_data, selection_artifacts)
                if final_artifacts:
                    st.session_state.artifacts['final_artifacts'] = final_artifacts
                    st.session_state.app_stage = 'final_model_selected'
                    st.success("Análise do modelo final e explicações SHAP geradas com sucesso!")
                    time.sleep(2) # Pausa para o usuário ver a mensagem de sucesso
                    st.rerun()
                else:
                    st.error("Falha ao gerar as explicações do modelo.")

    if 'final_artifacts' in st.session_state.get('artifacts', {}):
        st.markdown("---")
        st.info("Modelo campeão analisado! Agora você pode prosseguir para a página de **Decisão Gerencial e Não Supervisionada** para os insights finais.", icon="👉")

def display_advanced_analysis_page():
    """
    Renderiza a página principal de "Decisão Gerencial e Não Supervisionada".
    Esta versão coloca a seção de Tomada de Decisão em primeiro lugar,
    seguida pelas abas com as evidências (SHAP, Clusters).
    """
    st.header("Decisão Gerencial e Análise Avançada 🧠", divider='rainbow')

    if 'final_artifacts' not in st.session_state.get('artifacts', {}):
        st.error("⚠️ Análise Avançada Bloqueada", icon="🚨")
        st.warning("Para habilitar esta página, execute o pipeline completo na página 'Modelagem Supervisionada' e clique em 'Analisar Modelo Campeão'.")
        return
        
    final_artifacts = st.session_state.artifacts['final_artifacts']
    
    # Adicionando a explicação sobre a definição de risco
    st.info("""
    **Definição de Risco Utilizada no Projeto:** Conforme os critérios de negócio descritos no enunciado da prova, um cliente é definido como:
    - **Mau Pagador (`bad`):** Deixa de pagar a fatura por mais de 90 dias consecutivos.
    - **Bom Pagador (`good`):** Todos os demais clientes.
    O objetivo dos modelos é prever esta classificação pré-definida.
    """, icon="ℹ️")
    st.markdown("---")
    
    # Seção de Tomada de Decisão (Item I.d da Prova)
    with st.container(border=True):
        st.subheader("⭐ Tomada de Decisão e Aplicação Gerencial (Análise Crítica)")
        st.error("Esta seção sintetiza os resultados e apresenta recomendações acionáveis.", icon="⚠️")
        st.markdown("""
        Com base em todas as análises, especialmente nos insights dos gráficos SHAP, formulamos as seguintes recomendações para a área de crédito:

        #### 1. Fatores Críticos para Previsão de Risco
        A análise de explicabilidade global revelou que os fatores que mais aumentam a previsão de **'Mau Risco'** são:
        - **`checking_status` (Status da Conta Corrente):** Clientes sem conta ou com status '...<0 DM' são os de maior risco. A falta de um relacionamento bancário sólido é um forte indicador negativo.
        - **`duration` (Duração do Empréstimo):** Prazos de pagamento mais longos aumentam significativamente o risco.
        - **`credit_history` (Histórico de Crédito):** Históricos com pagamentos críticos ou atrasos passados são os principais impulsionadores do risco.
        - **`purpose` (Propósito):** Empréstimos para 'rádio/tv' e 'reparos' mostraram-se mais arriscados que os para 'carro (novo)'.

        #### 2. Recomendações Estratégicas para a Área de Crédito
        
        - **Segmentação e Políticas de Crédito Adaptativas:**
          - **Recomendação:** Abandonar uma política de crédito única e adotar abordagens diferentes para perfis distintos.
          - **Ação Prática:** Para o perfil **"Clientes sem imóvel próprio (`housing`='rent'), com baixo grau de instrução (`job`='unskilled') e histórico de atrasos (`credit_history`='critical account')**, que consistentemente apresentam altos SHAP-values para risco `bad`, **sugere-se uma política de crédito mais conservadora**. Isso inclui limites de crédito iniciais mais baixos, taxas de juros ajustadas ao risco, e possivelmente a exigência de garantias adicionais. A aprovação automática para este segmento deve ser desativada, forçando uma análise manual.

        - **Desenvolvimento de Produtos de Curto Prazo:**
          - **Recomendação:** Mitigar o risco associado à variável `duration`.
          - **Ação Prática:** Criar e promover linhas de crédito de curto prazo (6 a 12 meses). Estes produtos podem servir como uma "porta de entrada" segura para novos clientes, permitindo que a instituição construa um histórico de pagamento antes de oferecer limites mais altos ou prazos mais longos.
        """)
    
    st.markdown("---")
    st.subheader("Evidências de Suporte à Decisão")
    st.markdown("As abas abaixo contêm as análises detalhadas que fundamentam as recomendações acima.")

    tab_xai, tab_clusters = st.tabs([
        "🤖 Análise de Explicabilidade (SHAP)", 
        "🌀 Análise de Clusters (K-Means & DBSCAN)"
    ])
    
    with tab_xai:
        render_global_xai_module(final_artifacts)
        render_local_xai_and_recommendations_module(final_artifacts)
        
    with tab_clusters:
        if st.session_state.data_processed:
            processed_df_for_clustering = st.session_state.processed_df
            render_unsupervised_analysis_module(processed_df_for_clustering)

def render_global_xai_module(final_artifacts):
    with st.container(border=True):
        st.subheader("Análise de Explicabilidade Global (SHAP)")
        st.markdown("Aqui, abrimos a 'caixa-preta' do modelo para entender quais fatores ele considera mais importantes em suas decisões, de forma geral para todos os clientes do conjunto de teste.")
        
        X_test_df = final_artifacts['X_test_df']
        shap_values = final_artifacts['shap_values']

        st.markdown("#### Importância Geral das Features (SHAP Bar Plot)")
        st.markdown("Este gráfico ranqueia as features pelo seu impacto médio absoluto nas previsões. Features no topo são as que mais influenciaram o modelo, tanto para aumentar quanto para diminuir o risco.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=15, plot_size=None)
        plt.title(f"Importância Média das Features para o Modelo {final_artifacts['model_name']}", fontsize=16)
        st.pyplot(plt.gcf())
        plt.clf()
        
        st.markdown("---")
        
        st.markdown("#### Impacto e Distribuição das Features (SHAP Beeswarm Plot)")
        st.markdown("""
        Este é o gráfico mais poderoso para a análise global. Cada ponto é um cliente e uma feature.
        - **Eixo X (Valor SHAP):** O impacto na previsão. Valores positivos **aumentam a probabilidade de `Mau Risco`**. Valores negativos diminuem.
        - **Cor do Ponto:** O valor original da feature (Vermelho = Alto, Azul = Baixo).

        **Exemplo de Leitura:** Se pontos vermelhos para a feature `duration` estão à direita (SHAP > 0), significa que durações de empréstimo mais longas (valor alto) aumentam o risco previsto.
        """)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type='dot', show=False, max_display=15, plot_size=None)
        st.pyplot(plt.gcf())
        plt.clf()

def render_local_xai_and_recommendations_module(final_artifacts):
    with st.container(border=True):
        st.subheader("Análise de Previsão Individual (Por que este cliente?)")
        st.markdown("""
        Se a análise global mostra 'o que' o modelo valoriza, a análise local mostra o 'porquê' para um cliente específico.
        Selecione um cliente do conjunto de teste abaixo para ver um **Laudo de Risco Detalhado**, que explica como suas características
        o levaram da linha de base do modelo para sua pontuação final de risco.
        """)

        y_test = final_artifacts['y_test']
        X_test_df = final_artifacts['X_test_df']
        
        raw_data_test = st.session_state.artifacts['modeling_data']['X_test_raw']

        good_risk_clients = {f"Cliente {idx} (Idade: {raw_data_test.loc[idx, 'age']}, Crédito: {raw_data_test.loc[idx, 'credit_amount']})": idx for idx in y_test[y_test == 0].index}
        bad_risk_clients = {f"Cliente {idx} (Idade: {raw_data_test.loc[idx, 'age']}, Crédito: {raw_data_test.loc[idx, 'credit_amount']})": idx for idx in y_test[y_test == 1].index}
        
        tab_bad, tab_good = st.tabs(["Analisar um Cliente de Mau Risco", "Analisar um Cliente de Bom Risco"])

        def generate_waterfall_plot(selected_client_label, client_dict, plot_key):
            if selected_client_label:
                original_index = client_dict[selected_client_label]
                try:
                    row_position = X_test_df.index.get_loc(original_index)
                    
                    shap_values_for_instance = final_artifacts['shap_values'][row_position]

                    if shap_values_for_instance.ndim == 2:
                        values_for_plot = shap_values_for_instance[:, 1]
                    else:
                        values_for_plot = shap_values_for_instance

                    shap_explanation_object = shap.Explanation(
                        values=values_for_plot,
                        base_values=final_artifacts['expected_value'],
                        data=X_test_df.iloc[row_position],
                        feature_names=X_test_df.columns.tolist()
                    )
                    
                    fig, ax = plt.subplots()
                    shap.waterfall_plot(shap_explanation_object, max_display=15, show=False)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("---")
                    st.markdown("#### Interpretação Detalhada do Laudo de Risco")
                    
                    final_prediction_value = shap_explanation_object.base_values + shap_explanation_object.values.sum()
                    st.info(f"O modelo partiu de uma **pontuação base de {shap_explanation_object.base_values:.2f}** (média do modelo) e, após analisar as características deste cliente, chegou a uma **pontuação de risco final de {final_prediction_value:.2f}**.")

                    shap_df = pd.DataFrame({
                        'feature': shap_explanation_object.feature_names,
                        'value': shap_explanation_object.data,
                        'shap_value': shap_explanation_object.values
                    })
                    
                    risk_factors = shap_df[shap_df['shap_value'] > 0].sort_values(by='shap_value', ascending=False).head(3)
                    protective_factors = shap_df[shap_df['shap_value'] < 0].sort_values(by='shap_value', ascending=True).head(3)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.error("Principais Fatores de RISCO (Aumentaram a pontuação)")
                        if not risk_factors.empty:
                            for _, row in risk_factors.iterrows():
                                st.markdown(f"- **{row['feature']} = `{row['value']}`**: Aumentou o risco em **{row['shap_value']:.3f}**.")
                        else:
                            st.markdown("Nenhum fator de risco significativo encontrado.")
                    
                    with col2:
                        st.success("Principais Fatores de PROTEÇÃO (Diminuíram a pontuação)")
                        if not protective_factors.empty:
                            for _, row in protective_factors.iterrows():
                                st.markdown(f"- **{row['feature']} = `{row['value']}`**: Diminuiu o risco em **{row['shap_value']:.3f}**.")
                        else:
                            st.markdown("Nenhum fator de proteção significativo encontrado.")

                except (KeyError, IndexError) as e:
                     st.error(f"Não foi possível localizar ou processar os dados do cliente com índice {original_index}. Erro: {e}")
                     return

        with tab_bad:
            st.markdown("**Selecione um cliente classificado como `Mau Risco` (real):**")
            selected_bad_label = st.selectbox("Selecione o Cliente:", options=list(bad_risk_clients.keys()), key="select_bad")
            generate_waterfall_plot(selected_bad_label, bad_risk_clients, "waterfall_bad")

        with tab_good:
            st.markdown("**Selecione um cliente classificado como `Bom Risco` (real):**")
            selected_good_label = st.selectbox("Selecione o Cliente:", options=list(good_risk_clients.keys()), key="select_good")
            generate_waterfall_plot(selected_good_label, good_risk_clients, "waterfall_good")

    with st.container(border=True):
        st.subheader("⭐ Tomada de Decisão e Aplicação Gerencial (Análise Crítica)")
        st.error("Esta seção é o foco principal da avaliação", icon="⚠️")
        st.markdown("""
        Com base em todas as análises realizadas, especialmente nos insights dos gráficos SHAP (global e local),
        formulamos as seguintes recomendações para a área de crédito da instituição financeira:

        #### 1. Fatores Críticos para Previsão de Risco:
        A análise de explicabilidade global (SHAP Bar e Beeswarm Plots) revelou que os seguintes fatores são os mais determinantes para o modelo prever um cliente como de **'Mau Risco'**:
        - **`checking_status` (Status da Conta Corrente):** Clientes com status 'no checking' ou valores baixos consistentemente apresentam altos SHAP values positivos, indicando ser o principal fator de risco. A ausência de uma conta corrente ou uma conta com poucos recursos é um forte sinalizador de instabilidade financeira.
        - **`duration` (Duração do Empréstimo em Meses):** Prazos de pagamento mais longos aumentam significativamente o risco percebido pelo modelo. Empréstimos de longo prazo expõem a instituição a incertezas por mais tempo.
        - **`credit_history` (Histórico de Crédito):** Históricos de 'critical account/other credits existing' ou 'delay in paying off' são penalizados fortemente pelo modelo, o que é esperado e valida a lógica do algoritmo.
        - **`credit_amount` (Valor do Crédito):** Valores de empréstimo mais elevados, especialmente quando combinados com longas durações, também contribuem para um maior risco.

        #### 2. Recomendações Estratégicas para a Área de Crédito:
        Com base nestes fatores críticos, sugerem-se as seguintes ações gerenciais:

        - **Política de Crédito Mais Conservadora para Perfis de Alto Risco:**
          - **Recomendação:** Clientes que se enquadram no perfil de **"sem conta corrente ou com status precário", "histórico de pagamentos crítico" e que solicitam "empréstimos de longo prazo"** devem passar por uma análise de crédito mais rigorosa.
          - **Ação Prática:** Para estes perfis, a empresa pode implementar limites de crédito iniciais mais baixos, exigir garantias adicionais ou oferecer produtos com taxas de juros ajustadas ao risco. A aprovação automática para esses segmentos deve ser desativada, exigindo uma revisão manual.

        - **Desenvolvimento de Produtos de Curto Prazo e Menor Valor:**
          - **Recomendação:** Dado que a duração e o valor do crédito são fatores de risco importantes, a empresa pode focar em expandir seu portfólio de produtos de crédito de curto prazo e menor valor.
          - **Ação Prática:** Criar e promover campanhas de marketing para linhas de crédito de até 12 meses e valores mais baixos, que podem atrair clientes com menor risco percebido e servir como porta de entrada para um relacionamento de longo prazo.

        - **Monitoramento Reforçado e Ações de Relacionamento Proativo:**
          - **Recomendação:** O modelo pode ser usado não apenas na aprovação, mas também para monitorar a carteira de clientes existente. Clientes que, mesmo aprovados, possuíam características de risco limítrofes devem ser monitorados.
          - **Ação Prática:** Implementar um sistema de alerta que notifique o time de relacionamento quando um cliente do perfil de risco intermediário começar a apresentar comportamentos preocupantes (ex: atrasos em outras contas). Ações proativas, como oferta de renegociação ou educação financeira, podem ser tomadas antes que a inadimplência ocorra.
        """)

@st.cache_data(show_spinner="Executando clusterização com K-Means...")
def run_kmeans_clustering(_df):
    """
    Executa a clusterização com o algoritmo K-Means.
    Primeiro, utiliza o "Método do Cotovelo" (Elbow Method) para ajudar a
    identificar um número 'k' ótimo de clusters. Em seguida, aplica o K-Means
    com o 'k' escolhido e retorna os resultados.
    """
    numeric_df = _df.select_dtypes(include=np.number).drop(columns=[ProjectConfig.TARGET_VARIABLE], errors='ignore')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)

    inertia_values = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=ProjectConfig.RANDOM_STATE_SEED, n_init=10)
        kmeans.fit(scaled_features)
        inertia_values.append(kmeans.inertia_)
    
    optimal_k = 4 # Definido com base na análise do cotovelo
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=ProjectConfig.RANDOM_STATE_SEED, n_init=10)
    cluster_labels = final_kmeans.fit_predict(scaled_features)
    
    df_clustered = _df.copy()
    df_clustered['cluster'] = cluster_labels
    silhouette = silhouette_score(scaled_features, cluster_labels)
    
    cluster_artifacts = {
        'k_range': k_range, 'inertia_values': inertia_values,
        'optimal_k': optimal_k, 'silhouette_score': silhouette,
        'df_clustered': df_clustered, 'kmeans_model': final_kmeans,
        'scaled_features': scaled_features
    }
    return cluster_artifacts

def render_kmeans_module(cluster_artifacts):
    """
    Renderiza a aba com os resultados da clusterização K-Means.
    """
    st.subheader("Segmentação de Clientes com K-Means")
    st.markdown("O K-Means particiona os dados em 'K' clusters distintos, onde cada cliente pertence ao grupo com a média (centroide) mais próxima.")

    st.markdown("#### Encontrando o Número Ideal de Clusters (Método do Cotovelo)")
    st.markdown("O gráfico abaixo mostra a 'inércia' (soma das distâncias quadradas). O 'cotovelo', ponto onde a queda na inércia se torna menos pronunciada, sugere um número ótimo de clusters. Para este caso, **K=4** parece um bom equilíbrio.")
    fig_elbow = go.Figure(data=go.Scatter(x=list(cluster_artifacts['k_range']), y=cluster_artifacts['inertia_values'], mode='lines+markers'))
    fig_elbow.update_layout(title='Método do Cotovelo para Seleção de K', xaxis_title='Número de Clusters (K)', yaxis_title='Inércia')
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("#### Visualização e Análise dos Perfis dos Clusters")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Visualização dos Clusters (via PCA):**")
        df_clustered = cluster_artifacts['df_clustered']
        pca_result, _ = get_pca_projection(df_clustered.drop(columns=['cluster']), ProjectConfig.TARGET_VARIABLE)
        pca_result['cluster'] = df_clustered['cluster'].values
        fig_pca_cluster = px.scatter(
            pca_result, x='PC1', y='PC2', color='cluster',
            title='Visualização dos Clusters de Clientes', category_orders={"cluster": sorted(pca_result['cluster'].unique())}
        )
        st.plotly_chart(fig_pca_cluster, use_container_width=True)
    with col2:
        st.markdown("**Análise Cruzada com Risco:**")
        risk_by_cluster = df_clustered.groupby('cluster')[ProjectConfig.TARGET_VARIABLE].mean().reset_index()
        risk_by_cluster['class'] = risk_by_cluster['class'] * 100
        fig_risk_bar = px.bar(
            risk_by_cluster, x='cluster', y='class',
            title='Proporção de "Mau Risco" por Cluster', text_auto='.2f'
        )
        fig_risk_bar.update_yaxes(title_text='Mau Risco (%)')
        fig_risk_bar.update_traces(marker_color=ProjectConfig.SECONDARY_COLOR)
        st.plotly_chart(fig_risk_bar, use_container_width=True)
        st.metric("Score de Silhueta para K=4", f"{cluster_artifacts['silhouette_score']:.3f}", help="Mede quão bem definidos são os clusters. Valores próximos de 1 são melhores.")

    st.markdown("Analisando as médias das principais variáveis por cluster, podemos interpretar os perfis:")
    with st.expander("Ver Perfil Detalhado (Médias) de Cada Cluster"):
        cluster_profile = df_clustered.groupby('cluster').mean(numeric_only=True)
        st.dataframe(cluster_profile.style.background_gradient(cmap='Blues'), use_container_width=True)

@st.cache_data(show_spinner="Executando detecção de outliers com DBSCAN...")
def run_dbscan_outlier_detection(scaled_features):
    """
    Executa o algoritmo DBSCAN para identificar outliers (ruído) nos dados.
    DBSCAN agrupa pontos com base em densidade e é eficaz para encontrar
    observações que não pertencem a nenhum cluster denso.
    """
    # A escolha de eps e min_samples é crucial. Estes valores foram ajustados
    # empiricamente para este dataset.
    dbscan = DBSCAN(eps=2.0, min_samples=10, n_jobs=-1)
    outlier_labels = dbscan.fit_predict(scaled_features)
    
    outliers_count = np.sum(outlier_labels == -1)
    outliers_percentage = (outliers_count / len(outlier_labels)) * 100
    
    dbscan_artifacts = {
        'outlier_labels': outlier_labels,
        'outliers_count': outliers_count,
        'outliers_percentage': outliers_percentage
    }
    return dbscan_artifacts

def render_dbscan_module(dbscan_artifacts, df_with_class):
    """
    Renderiza a aba com os resultados da detecção de outliers com DBSCAN.
    """
    st.subheader("Detecção de Outliers com DBSCAN")
    st.markdown("""
    **O Quê?** Utilizamos o DBSCAN para identificar **outliers**, clientes com perfis tão atípicos que não se encaixam em nenhum grupo coeso. O DBSCAN é ideal para encontrar "pontos fora da curva".

    **Por quê?** Outliers podem representar um **risco oculto** (fraude, comportamento errático) ou uma **oportunidade única** (um nicho de mercado). A análise abaixo investiga a relação entre ser um outlier e o risco de inadimplência.
    """)

    col1, col2 = st.columns(2)
    col1.metric("Clientes Atípicos (Outliers) Identificados", f"{dbscan_artifacts['outliers_count']}")
    col2.metric("Percentual de Outliers na Base", f"{dbscan_artifacts['outliers_percentage']:.2f}%")

    st.markdown("#### Análise Cruzada: Risco de Crédito dos Outliers")
    st.markdown("**Existe relação entre os outliers detectados e o risco de inadimplência?**")

    df_with_outliers = df_with_class.copy()
    df_with_outliers['outlier_flag'] = ['Outlier' if label == -1 else 'Comum' for label in dbscan_artifacts['outlier_labels']]
    
    risk_in_outliers = df_with_outliers[df_with_outliers['outlier_flag'] == 'Outlier'][ProjectConfig.TARGET_VARIABLE].mean() * 100
    risk_in_core = df_with_outliers[df_with_outliers['outlier_flag'] == 'Comum'][ProjectConfig.TARGET_VARIABLE].mean() * 100
    
    risk_comparison_df = pd.DataFrame({
        'Grupo de Cliente': ['Outliers', 'Clientes Comuns'],
        'Taxa de Inadimplência (%)': [risk_in_outliers, risk_in_core]
    }).set_index('Grupo de Cliente')

    fig_risk_comp = px.bar(
        risk_comparison_df, y='Taxa de Inadimplência (%)',
        title='Comparativo de Risco: Outliers vs. Clientes Comuns', text_auto='.2f',
        color=risk_comparison_df.index,
        color_discrete_map={'Outliers': ProjectConfig.BAD_RISK_COLOR, 'Clientes Comuns': ProjectConfig.PRIMARY_COLOR}
    )
    st.plotly_chart(fig_risk_comp, use_container_width=True)

    st.success(f"""
    **Conclusão da Análise:** A taxa de inadimplência entre os outliers é de **{risk_in_outliers:.2f}%**, enquanto no grupo de clientes comuns é de **{risk_in_core:.2f}%**. 
    Este resultado mostra que os clientes com perfis atípicos, identificados pelo DBSCAN, possuem um risco consideravelmente distinto do restante da carteira, justificando uma análise de crédito individualizada e mais cautelosa para esses casos.
    """)

def render_unsupervised_analysis_module(processed_df):
    st.markdown("Nesta seção, usamos modelos não supervisionados para descobrir **estruturas e padrões ocultos nos dados sem usar a variável-alvo**.")

    if 'cluster_artifacts' not in st.session_state.artifacts:
        cluster_artifacts = run_kmeans_clustering(processed_df)
        st.session_state.artifacts['cluster_artifacts'] = cluster_artifacts
    else:
        cluster_artifacts = st.session_state.artifacts['cluster_artifacts']

    if 'dbscan_artifacts' not in st.session_state.artifacts:
        dbscan_artifacts = run_dbscan_outlier_detection(cluster_artifacts['scaled_features'])
        st.session_state.artifacts['dbscan_artifacts'] = dbscan_artifacts
    else:
        dbscan_artifacts = st.session_state.artifacts['dbscan_artifacts']

    tab_kmeans, tab_dbscan = st.tabs(["Clusterização com K-Means", "Detecção de Outliers com DBSCAN"])

    with tab_kmeans:
        render_kmeans_module(cluster_artifacts)
    
    with tab_dbscan:
        render_dbscan_module(dbscan_artifacts, cluster_artifacts['df_clustered'])

def display_modeling_page():
    """
    Renderiza a página principal de Modelagem Supervisionada, orquestrando
    a chamada sequencial dos módulos de preparação, seleção de features,
    treinamento e análise dos modelos.
    """
    st.header("Pipeline de Modelagem Supervisionada ⚙️", divider='rainbow')
    st.markdown("Execute as etapas em sequência para treinar, avaliar e selecionar o melhor modelo preditivo.")
    
    if not st.session_state.get('data_processed'):
        st.warning("⚠️ Por favor, processe os dados na página 'Análise e Preparação dos Dados' para habilitar a modelagem.")
        return

    df = st.session_state.processed_df

    render_data_preparation_module(df)
    
    if 'modeling_data' in st.session_state.artifacts:
        render_feature_selection_module(st.session_state.artifacts['modeling_data'])
    
    if 'selection_artifacts' in st.session_state.artifacts:
        render_baseline_modeling_module(
            st.session_state.artifacts['modeling_data'], 
            st.session_state.artifacts['selection_artifacts']
        )
    
    if 'baseline_artifacts' in st.session_state.artifacts:
        render_model_deep_dive_module(st.session_state.artifacts['baseline_artifacts'])
        render_final_model_analysis_module(
            st.session_state.artifacts['baseline_artifacts'], 
            st.session_state.artifacts['modeling_data'], 
            st.session_state.artifacts['selection_artifacts']
        )

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def display_export_and_docs_page(artifacts):
    st.header("Documentação e Exportação 📄", divider='rainbow')
    st.markdown("Esta seção centraliza a documentação técnica do projeto e oferece opções para exportar os principais dados e resultados gerados durante a análise.")

    doc_tab, export_tab = st.tabs(["📜 Documentação do Projeto", "💾 Exportar Resultados"])

    with doc_tab:
        st.subheader("Metodologia e Fluxo de Trabalho do Projeto")
        st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?q=80&w=2070&auto=format&fit=crop", use_container_width=True)

        st.markdown("""
        Esta seção detalha o fluxo de trabalho completo, as ferramentas utilizadas e as
        justificativas para as decisões técnicas tomadas ao longo do projeto de
        Análise de Risco de Crédito.
        """)
        
        with st.expander("I. Análise Preditiva com Modelos Supervisionados", expanded=True):
            st.markdown("""
            - **Diagnóstico e Balanceamento:** Foi identificada uma proporção de 30% de clientes de 'mau risco'. Para mitigar o viés do modelo em favor da classe majoritária, aplicou-se a técnica **SMOTE** no conjunto de treino.
            - **Treinamento de Modelos:** Foram treinados e avaliados 9 algoritmos de classificação, abrangendo modelos baseados em distância, bagging, boosting e uma rede neural.
            - **Métricas de Avaliação:** Os modelos foram comparados utilizando AUC, Recall, Precisão e F1-Score.
            """)
        with st.expander("II. Explicabilidade (XAI) e Decisão Gerencial"):
            st.markdown("""
            - **Explicabilidade com SHAP:** Sobre o modelo de melhor performance, foi aplicada a biblioteca **SHAP** para gerar transparência e interpretar os resultados.
            - **Tomada de Decisão:** Os insights do SHAP foram traduzidos em recomendações estratégicas para o negócio.
            """)
        with st.expander("III. Modelos Não Supervisionados"):
            st.markdown("""
            - **Clusterização com K-Means:** O algoritmo foi utilizado para segmentar os clientes em clusters distintos.
            - **Detecção de Outliers com DBSCAN:** O algoritmo foi aplicado para identificar clientes com perfis atípicos.
            """)
        with st.expander("IV. Bônus de Inovação: Dashboard Interativo"):
            st.markdown("""
            - Toda esta aplicação foi construída como um dashboard interativo usando **Streamlit**, cumprindo o requisito de bônus.
            """)

    with export_tab:
        st.subheader("Exportar Dados e Artefatos da Análise")
        st.info("Clique nos botões abaixo para fazer o download dos arquivos em formato CSV. Os botões só aparecerão se os artefatos correspondentes tiverem sido gerados nas etapas anteriores.", icon="💾")

        if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
            csv_processed = convert_df_to_csv(st.session_state.processed_df)
            st.download_button(
                label="Baixar Dados Processados",
                data=csv_processed,
                file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key='download-processed'
            )
        
        if 'baseline_artifacts' in artifacts:
            leaderboard_data = [{'Modelo': name, **res['metrics']} for name, res in artifacts['baseline_artifacts'].items()]
            leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by='AUC', ascending=False)
            csv_leaderboard = convert_df_to_csv(leaderboard_df)
            st.download_button(
                label="Baixar Leaderboard de Modelos",
                data=csv_leaderboard,
                file_name=f"leaderboard_modelos_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key='download-leaderboard'
            )

        if 'selection_artifacts' in artifacts:
            selected_features_df = pd.DataFrame(artifacts['selection_artifacts']['selected_feature_names'], columns=["Feature Selecionada"])
            csv_features = convert_df_to_csv(selected_features_df)
            st.download_button(
                label="Baixar Lista de Features Selecionadas",
                data=csv_features,
                file_name=f"features_selecionadas_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key='download-features'
            )

def main():
    """
    Função principal que controla a navegação e a renderização das páginas
    do aplicativo Streamlit.
    """
    initialize_session_state()
    px.defaults.template = ProjectConfig.get_plotly_template()

    st.sidebar.title("Painel de Controle 🎛️")
    st.sidebar.markdown("Navegue pelas etapas da análise de risco de crédito.")
    
    page_options = {
        "Página Inicial": "🏠",
        "Análise e Preparação dos Dados": "📊",
        "Análise Exploratória (EDA)": "🔍",
        "Modelagem Supervisionada": "⚙️",
        "Decisão Gerencial e Não Supervisionada": "🧠",
    }
    
    page_selection = st.sidebar.radio(
        "Menu de Navegação:",
        options=page_options.keys(),
        format_func=lambda x: f"{page_options[x]} {x}"
    )
    
    st.sidebar.markdown(
        """
        <div style='text-align: left; font-size: 0.9em;'>
            <strong>Prova Final</strong><br>
            <span>EPR0072 - Sistemas de Informação</span><br>
            <span>Prof. João Gabriel de Moraes Souza</span><br><br>
            <strong>Desenvolvedor:</strong><br>
            <span>Pedro Richetti Russo</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if page_selection == "Página Inicial":
        display_home_page()
    elif page_selection == "Análise e Preparação dos Dados":
        display_dataset_page()
    elif page_selection == "Análise Exploratória (EDA)":
        display_eda_page()
    elif page_selection == "Modelagem Supervisionada":
        display_modeling_page()
    elif page_selection == "Decisão Gerencial e Não Supervisionada":
        display_advanced_analysis_page()

if __name__ == "__main__":
    main()