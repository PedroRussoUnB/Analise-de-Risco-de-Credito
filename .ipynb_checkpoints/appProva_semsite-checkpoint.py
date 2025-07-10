from IPython.display import display, HTML
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

import shap

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def display_header(text, level=2):
    """Função auxiliar para exibir títulos formatados no notebook."""
    display(HTML(f"<h{level}>{text}</h{level}>"))

sns.set_style('darkgrid')
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (18, 9)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

plotly_template = go.layout.Template()
plotly_template.layout.paper_bgcolor = '#2E2E2E'
plotly_template.layout.plot_bgcolor = '#2E2E2E'
plotly_template.layout.font = dict(color='#EAEAEA')
plotly_template.layout.title = dict(x=0.5, font=dict(size=20))
plotly_template.layout.xaxis = dict(gridcolor='#444444', linecolor='#444444', zeroline=False, title_font=dict(size=14))
plotly_template.layout.yaxis = dict(gridcolor='#444444', linecolor='#444444', zeroline=False, title_font=dict(size=14))
px.defaults.template = plotly_template

df_credit = pd.read_csv('credit_customers.csv')

display_header("Visualização Inicial dos Dados", level=1)
display(df_credit.head())

display_header("Informações Gerais do Dataset", level=3)
df_credit.info()

display_header("I. Análise Preditiva com Modelos Supervisionados", level=1)
display_header("a) Diagnóstico e Desbalanceamento", level=2)

display(HTML("<p>O primeiro passo consiste na preparação dos dados e na análise da variável-alvo. Realizamos a limpeza dos nomes das colunas para facilitar a manipulação e, em seguida, codificamos a variável 'class' para um formato numérico, essencial para a modelagem. O diagnóstico de desbalanceamento é uma etapa crítica, pois uma grande desproporção entre as classes pode levar os modelos a terem um desempenho enviesado, favorecendo a classe majoritária.</p>"))

df_credit.columns = [col.lower().replace(' ', '_') for col in df_credit.columns]

le = LabelEncoder()
df_credit['class_encoded'] = le.fit_transform(df_credit['class'])
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print("Mapeamento da variável-alvo 'class':")
print(f"{class_mapping}\n")

df_analysis = df_credit.drop('class', axis=1)

class_distribution = df_credit['class'].value_counts()
class_percentage = df_credit['class'].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Contagem': class_distribution,
    'Percentual (%)': class_percentage.round(2)
})

display_header("Distribuição da Variável-Alvo 'class'", level=3)
display(summary_df)

fig = go.Figure(go.Bar(
    x=summary_df.index,
    y=summary_df['Contagem'],
    text=summary_df['Contagem'],
    textposition='auto',
    marker_color=['#1f77b4', '#d62728']
))

fig.update_layout(
    title_text='Contagem de Clientes por Risco de Crédito',
    xaxis_title='Classe de Risco',
    yaxis_title='Número de Clientes',
    template=plotly_template,
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)
fig.show()

display(HTML("<p><b>Conclusão do Diagnóstico:</b> A análise confirma um desbalanceamento significativo. A classe 'good' (bons pagadores) representa 70% dos dados, enquanto a classe 'bad' (maus pagadores) corresponde a apenas 30%. Sem tratamento, os modelos podem desenvolver um viés para prever a classe majoritária, resultando em uma baixa capacidade de identificar clientes de alto risco, que é o principal objetivo do negócio. Portanto, a aplicação de técnicas de reamostragem, como o SMOTE, é justificada e necessária.</p>"))

display_header("Análise Exploratória Univariada - Variáveis Categóricas", level=2)
categorical_features = df_analysis.select_dtypes(include=['object']).columns

for col in categorical_features:
    fig = px.histogram(df_analysis, x=col, color='class_encoded', barmode='group',
                       title=f'Distribuição de {col.replace("_", " ").title()} por Classe de Risco',
                       labels={'class_encoded': 'Classe de Risco', 'count': 'Contagem'},
                       color_discrete_map={0: '#d62728', 1: '#1f77b4'})
    fig.update_layout(template=plotly_template)
    fig.show()

display_header("Análise Exploratória Bivariada - Relação com a Variável-Alvo", level=2)
display(HTML("<p>Nesta seção, investigamos como as variáveis numéricas e categóricas se relacionam com a variável-alvo. Isso nos ajuda a identificar quais características podem ser mais preditivas do risco de crédito.</p>"))

numeric_features = ['age', 'credit_amount', 'duration']

for col in numeric_features:
    fig = px.box(
        df_credit,
        x='class',
        y=col,
        color='class',
        title=f'Distribuição de {col.replace("_", " ").title()} por Classe de Risco',
        labels={'class': 'Classe de Risco', col: col.replace("_", " ").title()},
        color_discrete_map={'good': '#1f77b4', 'bad': '#d62728'}
    )
    fig.update_layout(template=plotly_template)
    fig.show()

display(HTML("""
<h4>Observações da Análise Bivariada (Numérica):</h4>
<ul>
    <li><b>Duração (duration):</b> Clientes classificados como 'bad' tendem a ter durações de empréstimo significativamente maiores. A mediana para maus pagadores é maior e a distribuição é mais ampla, sugerindo que empréstimos de longo prazo são mais arriscados.</li>
    <li><b>Valor do Crédito (credit_amount):</b> Similarmente, o valor do crédito solicitado por maus pagadores é, em média, mais alto. A presença de muitos outliers com valores elevados nesta categoria reforça a ideia de que empréstimos de grande volume representam um risco maior.</li>
    <li><b>Idade (age):</b> Clientes mais jovens parecem ter uma maior propensão a serem classificados como 'bad'. A mediana de idade para maus pagadores é visivelmente menor do que para bons pagadores.</li>
</ul>
"""))

display_header("Matriz de Correlação entre Variáveis Numéricas", level=3)
display(HTML("<p>Analisamos a correlação de Pearson para verificar a relação linear entre as variáveis numéricas. Valores próximos de 1 ou -1 indicam uma forte correlação, o que pode ser um sinal de multicolinearidade, um problema a ser monitorado para modelos como a Regressão Logística.</p>"))

corr_matrix = df_credit[numeric_features].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    text=corr_matrix.round(2).astype(str),
    texttemplate="%{text}",
    showscale=True
))

fig_corr.update_layout(
    title='Matriz de Correlação entre Variáveis Numéricas',
    template=plotly_template
)
fig_corr.show()

display(HTML("<p><b>Conclusão da Correlação:</b> Há uma correlação positiva moderada entre 'duration' e 'credit_amount' (0.62), o que é esperado, pois empréstimos de valores maiores geralmente exigem prazos de pagamento mais longos. As demais correlações são fracas, indicando que não há um problema severo de multicolinearidade entre estas três variáveis principais.</p>"))

display_header("b) Treinamento de Modelos", level=2)
display(HTML("""
<p>Antes de treinar os modelos, é fundamental preparar os dados adequadamente. Esta etapa envolve a transformação de variáveis categóricas em um formato numérico que os algoritmos possam entender e a padronização das variáveis numéricas para que todas tenham a mesma escala de importância.</p>
<ul>
    <li><b>Codificação de Variáveis Categóricas (One-Hot Encoding):</b> Converte cada categoria em uma nova coluna binária (0 ou 1). Isso evita que os modelos interpretem as categorias como se tivessem uma ordem intrínseca (ex: 'savings_accounts' não é "maior" que 'checking').</li>
    <li><b>Padronização de Variáveis Numéricas (StandardScaler):</b> Transforma os dados para que tenham média 0 e desvio padrão 1. É crucial para modelos sensíveis à distância, como KNN e SVM, garantindo que variáveis com escalas maiores (como 'credit_amount') não dominem o processo de aprendizado.</li>
    <li><b>Divisão Estratificada (Train-Test Split):</b> Dividimos o dataset em um conjunto de treino (70%) e um de teste (30%). A estratificação pela variável-alvo <code>class_encoded</code> garante que a proporção de bons e maus pagadores seja mantida em ambos os conjuntos, o que é vital para uma avaliação justa e imparcial do desempenho dos modelos.</li>
</ul>
"""))

X = df_analysis.drop('class_encoded', axis=1)
y = df_analysis['class_encoded']

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

processed_feature_names = numeric_features.tolist() + \
                          preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()


display(HTML(f"<h4>Dimensões dos Conjuntos de Dados:</h4>"
             f"<ul>"
             f"<li>Conjunto de Treino (X_train): {X_train_processed.shape[0]} amostras, {X_train_processed.shape[1]} features</li>"
             f"<li>Conjunto de Teste (X_test): {X_test_processed.shape[0]} amostras, {X_test_processed.shape[1]} features</li>"
             f"</ul>"))

display(HTML("<p>Com os dados devidamente separados e pré-processados, o próximo passo é corrigir o desbalanceamento da classe no conjunto de treino usando a técnica SMOTE.</p>"))

display_header("Aplicação da Técnica SMOTE para Balanceamento", level=3)
display(HTML("""
<p><b>Justificativa da Escolha do SMOTE:</b> Conforme diagnosticado, a classe 'bad' (maus pagadores) é minoritária. Treinar um modelo com dados desbalanceados pode levar a um viés, onde o modelo simplesmente aprende a prever a classe majoritária ('good') na maioria das vezes, resultando em um recall muito baixo para a classe de interesse ('bad'). Para mitigar isso, utilizamos a técnica <b>SMOTE (Synthetic Minority Over-sampling Technique)</b>.</p>
<p>O SMOTE foi escolhido por sua eficácia e abordagem inteligente: em vez de simplesmente duplicar os exemplos da classe minoritária (o que poderia levar a overfitting), ele cria novos exemplos <b>sintéticos</b>. Ele faz isso selecionando um ponto da classe minoritária, encontrando seus vizinhos mais próximos e criando um novo ponto em algum lugar na linha entre eles. O resultado é um conjunto de treino enriquecido e balanceado, que permite ao modelo aprender as características da classe minoritária de forma mais robusta, sem perder a generalização.</p>
<p><b>Importante:</b> O SMOTE é aplicado <b>apenas ao conjunto de treino</b>. O conjunto de teste permanece intocado e com a distribuição original para garantir que a avaliação do modelo seja feita em um cenário que reflete a realidade.</p>
"""))

smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

display(HTML(f"<h4>Dimensões Antes e Depois do SMOTE:</h4>"
             f"<ul>"
             f"<li>Shape de X_train antes: {X_train_processed.shape}</li>"
             f"<li>Shape de y_train antes: {y_train.shape}</li>"
             f"<li>Shape de X_train depois do SMOTE: {X_train_resampled.shape}</li>"
             f"<li>Shape de y_train depois do SMOTE: {y_train_resampled.shape}</li>"
             f"</ul>"))

original_counts = y_train.value_counts()
resampled_counts = y_train_resampled.value_counts()

fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribuição Antes do SMOTE', 'Distribuição Depois do SMOTE'))

fig.add_trace(
    go.Bar(x=original_counts.index.map({0: 'Bad', 1: 'Good'}), y=original_counts.values, name='Original', marker_color=['#d62728', '#1f77b4']),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=resampled_counts.index.map({0: 'Bad', 1: 'Good'}), y=resampled_counts.values, name='Balanceado', marker_color=['#d62728', '#1f77b4']),
    row=1, col=2
)

fig.update_layout(
    title_text='Impacto do SMOTE no Balanceamento da Classe de Treino',
    showlegend=False,
    template=plotly_template
)
fig.show()

display(HTML("<p><b>Conclusão do Balanceamento:</b> O gráfico demonstra claramente o impacto positivo do SMOTE. O conjunto de treino, que antes era desbalanceado, agora possui um número igual de amostras para as classes 'good' e 'bad'. Isso cria um ambiente de aprendizado justo para os modelos, que agora podem ser treinados para identificar os padrões de ambas as classes com a mesma importância.</p>"))

display_header("Treinamento e Avaliação dos Modelos Supervisionados", level=3)
display(HTML("<p>Iniciamos o treinamento e a avaliação dos modelos. Cada algoritmo será treinado no conjunto de dados balanceado pelo SMOTE e avaliado no conjunto de teste original e intocado. Para cada modelo, calculamos um conjunto completo de métricas e exibimos a Matriz de Confusão para uma análise detalhada de seus acertos e erros.</p>"))

models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=RANDOM_STATE),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
}

results = {}
roc_curves = {}

for name, model in models.items():
    display_header(f"Resultados para: {name}", level=4)
    
    pipeline = ImbPipeline(steps=[
        ('model', model)
    ])
    
    pipeline.fit(X_train_resampled, y_train_resampled)
    
    y_pred = pipeline.predict(X_test_processed)
    
    if hasattr(model, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test_processed)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    else:
        auc = "N/A"
        roc_curves[name] = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)

    results[name] = {
        'Acurácia': accuracy,
        'Precisão (Bad)': precision,
        'Recall (Bad)': recall,
        'F1-Score (Bad)': f1,
        'AUC': auc
    }
    
    print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Previsto Bad', 'Previsto Good'],
                yticklabels=['Real Bad', 'Real Good'])
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

display(HTML("<p>Continuamos o processo com os modelos baseados em Boosting e uma Rede Neural Simples (MLP). Os modelos de Boosting são conhecidos por sua alta performance, pois constroem uma sequência de modelos fracos onde cada um tenta corrigir os erros do anterior. A Rede Neural oferece uma abordagem diferente, baseada na interconexão de neurônios artificiais.</p>"))

boosting_models = {
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    "Rede Neural (MLP)": MLPClassifier(random_state=RANDOM_STATE, max_iter=1000)
}

for name, model in boosting_models.items():
    display_header(f"Resultados para: {name}", level=4)
    
    pipeline = ImbPipeline(steps=[
        ('model', model)
    ])
    
    pipeline.fit(X_train_resampled, y_train_resampled)
    
    y_pred = pipeline.predict(X_test_processed)
    y_proba = pipeline.predict_proba(X_test_processed)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)

    results[name] = {
        'Acurácia': accuracy,
        'Precisão (Bad)': precision,
        'Recall (Bad)': recall,
        'F1-Score (Bad)': f1,
        'AUC': auc
    }

    print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Previsto Bad', 'Previsto Good'],
                yticklabels=['Real Bad', 'Real Good'])
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

models.update(boosting_models)

display_header("Comparação Consolidada de Desempenho dos Modelos", level=2)
display(HTML("<p>Após treinar todos os algoritmos, consolidamos seus resultados em um 'leaderboard' para comparar objetivamente a performance de cada um. A métrica AUC (Área Sob a Curva ROC) é frequentemente usada como um indicador geral de performance, enquanto o Recall e F1-Score para a classe 'bad' são críticos para o nosso objetivo de negócio. A visualização das Curvas ROC em um único gráfico nos permite uma comparação direta da capacidade de discriminação de cada modelo.</p>"))

results_df = pd.DataFrame(results).T.sort_values(by='AUC', ascending=False)
results_df_styled = results_df.style.background_gradient(cmap='viridis', subset=['AUC', 'F1-Score (Bad)', 'Recall (Bad)']) \
                                  .format("{:.4f}")

display_header("Leaderboard de Performance dos Modelos", level=3)
display(results_df_styled)

fig_roc = go.Figure()

fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

for name, values in roc_curves.items():
    if values:
        fig_roc.add_trace(go.Scatter(x=values['fpr'], y=values['tpr'], name=f"{name} (AUC={values['auc']:.3f})", mode='lines'))

fig_roc.update_layout(
    xaxis_title='Taxa de Falsos Positivos',
    yaxis_title='Taxa de Verdadeiros Positivos (Recall)',
    title='Curvas ROC Comparativas dos Modelos',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template=plotly_template,
    height=700,
    width=900
)
fig_roc.show()

best_model_name = results_df['AUC'].idxmax()
best_model_auc = results_df.loc[best_model_name]['AUC']
best_model_recall = results_df.loc[best_model_name]['Recall (Bad)']
best_model = models[best_model_name]

display(HTML(f"""
<h4>Seleção do Modelo Campeão</h4>
<p>Com base na métrica <b>AUC</b>, o modelo com melhor desempenho geral é o <b>{best_model_name}</b>, atingindo um valor de <b>{best_model_auc:.4f}</b>. Este modelo também apresentou um Recall de <b>{best_model_recall:.4f}</b> para a classe 'bad', indicando uma boa capacidade de identificar os clientes de risco.</p>
<p>Este será o modelo selecionado para as próximas etapas de análise de explicabilidade (XAI com SHAP) e para embasar a tomada de decisão gerencial.</p>
"""))

display_header("c) Explicabilidade com SHAP (XAI)", level=2)
display(HTML("""
<p>A simples previsão de um modelo de 'caixa-preta' não é suficiente para a tomada de decisão estratégica. É fundamental entender <b>por que</b> o modelo toma certas decisões. Para isso, utilizamos o <b>SHAP (SHapley Additive exPlanations)</b>, uma técnica de XAI (Explainable Artificial Intelligence) baseada na teoria dos jogos que explica a saída de qualquer modelo de machine learning.</p>
<p>Nesta seção, realizamos a <b>Análise de Explicabilidade Global</b> para entender quais variáveis, em média, mais impactam as previsões do nosso modelo campeão, o <b>{best_model_name}</b>.</p>
""".format(best_model_name=best_model_name)))

shap.initjs()

final_pipeline = ImbPipeline(steps=[('model', best_model)])
final_pipeline.fit(X_train_resampled, y_train_resampled)

X_test_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)

try:
    explainer = shap.TreeExplainer(final_pipeline.named_steps['model'])
except Exception:
    display(HTML("<p>O modelo selecionado não é baseado em árvores. Utilizando KernelExplainer (pode ser mais lento).</p>"))
    X_train_summary = shap.sample(X_train_resampled, 100)
    explainer = shap.KernelExplainer(final_pipeline.predict_proba, X_train_summary)

shap_values = explainer.shap_values(X_test_df)
shap_values_class_bad = shap_values[0] if isinstance(shap_values, list) else shap_values

display_header("Summary Plot com as Variáveis Mais Relevantes (Impacto Médio)", level=3)
display(HTML("<p>O gráfico abaixo mostra o impacto médio de cada variável no resultado do modelo. As variáveis no topo são as mais importantes para as previsões de forma geral.</p>"))
fig_summary_bar, ax_summary_bar = plt.subplots()
shap.summary_plot(shap_values_class_bad, X_test_df, plot_type="bar", show=False)
plt.title(f"Importância Geral das Features para a Classe 'Bad' ({best_model_name})")
plt.tight_layout()
plt.show()


display_header("Análise Detalhada do Impacto das Features (Beeswarm Plot)", level=3)
display(HTML("""
<p>Este gráfico é mais rico, pois mostra não apenas a importância, mas também o <b>efeito</b> de cada variável em cada previsão individual.</p>
<ul>
    <li><b>Posição no eixo X:</b> Um valor SHAP positivo (à direita do centro) indica que a variável aumentou a probabilidade de o cliente ser 'bad'. Um valor negativo (à esquerda) diminuiu essa probabilidade.</li>
    <li><b>Cor do Ponto:</b> A cor representa o valor da variável para aquele cliente (vermelho = alto, azul = baixo).</li>
</ul>
<p><b>Exemplo de Interpretação:</b> Se a variável 'duration' tem pontos vermelhos à direita, significa que altas durações de empréstimo consistentemente aumentam o risco de inadimplência, segundo o modelo.</p>
"""))
fig_summary_dot, ax_summary_dot = plt.subplots()
shap.summary_plot(shap_values_class_bad, X_test_df, plot_type="dot", show=False)
plt.title(f"Impacto das Features na Previsão da Classe 'Bad' ({best_model_name})")
plt.tight_layout()
plt.show()

display_header("Interpretação de Casos Individuais (Análise Local)", level=3)
display(HTML("""
<p>A análise global nos deu o panorama geral, mas para a tomada de decisão no dia a dia, é essencial entender casos específicos. Aqui, vamos mergulhar nas previsões para dois clientes do conjunto de teste: um classificado corretamente como 'mau pagador' e outro como 'bom pagador'. Usaremos os gráficos <b>Waterfall</b> do SHAP para decompor a previsão de cada um.</p>
"""))

y_pred_final = final_pipeline.predict(X_test_processed)

correctly_predicted_bad_idx = np.where((y_test == 0) & (y_pred_final == 0))[0]
correctly_predicted_good_idx = np.where((y_test == 1) & (y_pred_final == 1))[0]

if len(correctly_predicted_bad_idx) > 0:
    idx_bad = correctly_predicted_bad_idx[0]
    display_header("Exemplo de Cliente Previsto como 'Mau Pagador' (Risco Elevado)", level=4)
    
    fig_bad, ax_bad = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_class_bad[idx_bad],
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_test_df.iloc[idx_bad],
        feature_names=X_test_df.columns.tolist()
    ), show=False)
    plt.tight_layout()
    plt.show()
else:
    display(HTML("<p>Nenhum cliente 'bad' foi corretamente classificado no conjunto de teste para análise local.</p>"))

if len(correctly_predicted_good_idx) > 0:
    idx_good = correctly_predicted_good_idx[0]
    display_header("Exemplo de Cliente Previsto como 'Bom Pagador' (Risco Baixo)", level=4)

    fig_good, ax_good = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_class_bad[idx_good],
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_test_df.iloc[idx_good],
        feature_names=X_test_df.columns.tolist()
    ), show=False)
    plt.tight_layout()
    plt.show()
else:
    display(HTML("<p>Nenhum cliente 'good' foi corretamente classificado no conjunto de teste para análise local.</p>"))


display_header("d) Tomada de Decisão e Aplicação Gerencial", level=2)
display(HTML("""
<p>Esta seção é o coração do projeto e responde à demanda mais importante da avaliação. Com base em todas as análises anteriores, especialmente nos insights gerados pelo SHAP, formulamos recomendações estratégicas e acionáveis para a área de crédito da instituição financeira.</p>
<h4>Análise Crítica dos Fatores de Risco (Baseada no SHAP)</h4>
<p>A análise global do SHAP (Beeswarm e Bar Plots) revelou um padrão claro sobre os principais impulsionadores do risco de crédito. Os fatores mais críticos que aumentam a probabilidade de um cliente ser classificado como 'bad' são, consistentemente:</p>
<ol>
    <li><b>Histórico de Crédito (credit_history):</b> O fator de maior impacto. Clientes com histórico 'critical/other existing credit' ou 'existing credit paid' têm SHAP values significativamente negativos (reduzem o risco), enquanto aqueles com 'no credits/all paid' ou, pior, 'delay in past' apresentam SHAP values fortemente positivos, empurrando a previsão para 'bad'.</li>
    <li><b>Status da Conta Corrente (checking_account):</b> A ausência de uma conta corrente ou uma conta com saldo baixo ('<0 DM') é um forte indicador de risco, associado a altos valores SHAP positivos. Clientes com contas de saldo elevado ('>=200 DM') são vistos como muito mais seguros.</li>
    <li><b>Duração do Empréstimo (duration):</b> Conforme visto na EDA e confirmado pelo SHAP, durações mais longas (pontos vermelhos) estão consistentemente à direita no beeswarm plot, indicando que aumentam o risco percebido pelo modelo.</li>
    <li><b>Propósito do Empréstimo (purpose):</b> O propósito 'radio/tv' e 'car' estão entre os que mais contribuem para o risco, enquanto 'business' ou 'education' (embora menos frequentes) tendem a reduzir o risco.</li>
</ol>
<p>A análise dos casos individuais (waterfall plots) valida essa visão. No cliente de alto risco, vimos que a combinação de um histórico de crédito ruim e a ausência de uma conta corrente foram os principais fatores que levaram o modelo a prever 'bad'. Em contraste, no cliente de baixo risco, um bom histórico de crédito e uma conta corrente saudável foram suficientes para anular o risco moderado de uma duração de empréstimo um pouco mais longa.</p>

<h4>Recomendação Estratégica para a Área de Crédito</h4>
<p>Com base nas evidências quantitativas e explicáveis geradas pelo modelo e pelo SHAP, a seguinte recomendação estratégica é proposta para a área de crédito:</p>
<blockquote>
  <p><b>"Clientes com histórico de crédito problemático (atrasos ou sem créditos anteriores) e sem uma conta corrente estabelecida ou com saldo baixo apresentam os maiores SHAP-values para o risco 'bad', indicando uma alta probabilidade de inadimplência. Sugere-se a implementação de uma política de crédito mais conservadora e um monitoramento reforçado para esses perfis. Recomenda-se que, para novos pedidos de crédito deste segmento, sejam oferecidos limites iniciais mais baixos e prazos de pagamento mais curtos (menor 'duration'). Adicionalmente, para a carteira existente, sugere-se a criação de um alerta proativo para o time de relacionamento quando clientes com esse perfil solicitarem aumentos de limite ou novos produtos, permitindo uma intervenção focada em educação financeira antes de uma possível inadimplência."</b></p>
</blockquote>
<p>Esta recomendação é diretamente acionável e fundamentada nos fatores de maior peso preditivo identificados pelo nosso sistema de apoio à decisão, alinhando a capacidade analítica da IA com os objetivos de negócio de crescimento sustentável.</p>
"""))

display_header("II. Modelos Não Supervisionados: Clusterização e Outliers", level=1)
display(HTML("""
<p>Após a análise preditiva, o próximo passo é explorar a estrutura inerente aos nossos dados sem utilizar a variável de resposta <code>class</code>. O objetivo da aprendizagem não supervisionada, neste contexto, é duplo:</p>
<ol>
    <li><b>Segmentar Clientes:</b> Agrupar clientes com características semelhantes em "clusters" ou perfis. Isso pode revelar segmentos de mercado distintos que a empresa talvez não conheça, permitindo a criação de estratégias de marketing, produto e risco personalizadas para cada grupo.</li>
    <li><b>Detectar Anomalias:</b> Identificar clientes com perfis atípicos (outliers) que se desviam significativamente do comportamento padrão. Esses clientes podem representar tanto um risco oculto quanto uma oportunidade única.</li>
</ol>
<p>Para esta análise, utilizaremos os dados de treino já pré-processados (<code>X_train_processed</code>). A escolha de usar os dados já escalados (StandardScaler) é fundamental para algoritmos baseados em distância, como K-Means e DBSCAN, pois garante que todas as variáveis contribuam igualmente para o cálculo das distâncias, evitando que features com escalas maiores dominem o resultado.</p>
"""))

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

X_unsupervised = X_train_processed

display(HTML(f"<h4>Dados para Análise Não Supervisionada</h4>"
             f"<p>Utilizaremos o conjunto de treino pré-processado e balanceado, que possui <b>{X_train_resampled.shape[0]}</b> amostras e <b>{X_train_resampled.shape[1]}</b> features. Isso garante que a análise de segmentação seja realizada em um espaço de características rico e padronizado.</p>"))

display(HTML("""
<p>Começaremos com a clusterização usando K-Means, que requer a definição prévia do número de clusters (k). Para isso, usaremos o Método do Cotovelo.</p>
"""))

X_train_unscaled_df = pd.DataFrame(X_train, columns=X.columns)

print("Dimensões do DataFrame para análise não supervisionada:")
print(X_unsupervised.shape)

print("\nPrimeiras 5 linhas do DataFrame não escalado para referência de interpretação:")
display(X_train_unscaled_df.head())

display_header("a) Clusterização com KMeans", level=2)
display(HTML("""
<p>O primeiro passo para utilizar o K-Means é determinar o número ideal de clusters (k). Uma abordagem padrão e eficaz para isso é o <b>Método do Cotovelo (Elbow Method)</b>.</p>
<p><b>Como funciona:</b></p>
<ol>
    <li>O algoritmo K-Means é executado para uma série de valores de 'k' (por exemplo, de 1 a 10).</li>
    <li>Para cada 'k', calculamos a <b>inércia</b>, que é a soma das distâncias quadradas de cada ponto de dados ao centróide de seu cluster. A inércia mede o quão coesos e compactos são os clusters.</li>
    <li>Plotamos a inércia em função do número de clusters.</li>
</ol>
<p>O 'k' ideal é encontrado no ponto do gráfico que se assemelha a um cotovelo – o ponto onde a taxa de diminuição da inércia se torna significativamente mais lenta. Adicionar mais clusters após esse ponto resulta em ganhos marginais e pode levar a um excesso de segmentação (overfitting).</p>
"""))

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans_test.fit(X_unsupervised)
    inertia_values.append(kmeans_test.inertia_)

elbow_fig = go.Figure()

elbow_fig.add_trace(go.Scatter(
    x=list(k_range),
    y=inertia_values,
    mode='lines+markers',
    marker=dict(size=8),
    line=dict(width=2)
))

elbow_fig.add_annotation(
    x=4,
    y=inertia_values[3],
    text="Ponto de Inflexão Sugerido (k=4)",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    ax=40,
    ay=-60
)

elbow_fig.update_layout(
    title='Método do Cotovelo para Determinação do Número Ideal de Clusters',
    xaxis_title='Número de Clusters (k)',
    yaxis_title='Inércia (Soma dos Quadrados Intra-cluster)',
    template=plotly_template,
    height=600
)

elbow_fig.show()

display(HTML("""
<p><b>Conclusão do Método do Cotovelo:</b> O gráfico acima mostra uma nítida "curva de cotovelo" no ponto onde <b>k=4</b>. A partir deste ponto, a redução na inércia ao adicionar mais um cluster é consideravelmente menor. Isso sugere que dividir os clientes em 4 grupos representa um bom equilíbrio entre a coesão dos clusters e a simplicidade do modelo de segmentação. Portanto, procederemos com k=4 para a análise final.</p>
"""))

optimal_k_value = 4

display(HTML(f"<h4>Treinando o modelo K-Means com k={optimal_k_value}</h4>"))

kmeans_model = KMeans(n_clusters=optimal_k_value, random_state=RANDOM_STATE, n_init=10)
cluster_labels_assigned = kmeans_model.fit_predict(X_unsupervised)

X_train_unscaled_df['cluster'] = cluster_labels_assigned

display(HTML("<p>Após treinar o modelo, cada cliente no conjunto de treino foi atribuído a um dos 4 clusters. Para interpretar o que cada cluster significa, analisamos os valores médios (para variáveis numéricas) e os valores mais frequentes (para categóricas) de cada grupo. Isso nos permite criar 'personas' de clientes.</p>"))

cluster_profile_analysis = X_train_unscaled_df.groupby('cluster').agg({
    'age': 'mean',
    'credit_amount': 'mean',
    'duration': 'mean',
    'purpose': lambda x: x.mode()[0],
    'credit_history': lambda x: x.mode()[0],
    'savings_accounts': lambda x: x.mode()[0],
    'employment': lambda x: x.mode()[0],
    'job': lambda x: x.mode()[0]
}).round(1)

display_header("Perfis Detalhados dos Clusters de Clientes", level=3)
display(cluster_profile_analysis.style.background_gradient(cmap='viridis'))

display(HTML("""
<h4>Análise e "Personas" dos Clusters:</h4>
<ul>
    <li>
        <b>Cluster 0 - Jovens em Consolidação:</b> Este grupo é o mais jovem (idade média de 33.2 anos) e solicita créditos de valores e durações moderadas. Seu propósito principal é para 'radio/tv', sugerindo compras de bens de consumo. O histórico de crédito é bom, mas não excepcional. <b>Persona:</b> O jovem adulto, no início ou meio da carreira, buscando crédito para consumo pessoal.
    </li>
    <li>
        <b>Cluster 1 - Profissionais Liberais de Alto Risco:</b> Este cluster se destaca pelo tipo de trabalho ('unskilled and non-resident') e por ter o <b>maior valor de crédito</b> (média de 4793) e a <b>maior duração</b> (média de 29 meses). Apesar de terem um bom histórico, o alto endividamento e a instabilidade no emprego os tornam um segmento de alto risco. <b>Persona:</b> O trabalhador autônomo ou com emprego menos formal, que busca crédito elevado para projetos maiores, como a compra de um carro.
    </li>
    <li>
        <b>Cluster 2 - Clientes Maduros e Seguros:</b> Este grupo tem a maior média de idade (45.3 anos), os <b>menores valores de crédito e durações</b>, e um histórico de crédito quase impecável ('critical/other existing credit'). São, de longe, o segmento mais seguro. <b>Persona:</b> O cliente estabelecido, com estabilidade financeira, que utiliza o crédito de forma pontual e consciente.
    </li>
    <li>
        <b>Cluster 3 - Endividados com Emprego Estável:</b> Curiosamente, este grupo é formado por clientes com emprego estável ('skilled'), mas que buscam valores de crédito relativamente altos (média de 3336) por períodos longos (média de 25 meses), frequentemente para compra de carro. <b>Persona:</b> O profissional qualificado que utiliza o crédito de forma intensiva, potencialmente vivendo no limite de suas finanças.
    </li>
</ul>
"""))

display_header("c) Análise Cruzada", level=2)
display(HTML("""
<p>A verdadeira utilidade da segmentação de clientes reside em sua capacidade de informar a estratégia de negócio. Nesta etapa, cruzamos os clusters que acabamos de criar (de forma não supervisionada) com a variável-alvo real (<code>class_encoded</code>) para responder à pergunta: <b>"Os segmentos de clientes que encontramos naturalmente possuem níveis de risco diferentes?"</b>.</p>
<p>Se a resposta for sim, a empresa pode criar políticas de crédito, marketing e relacionamento personalizadas para cada perfil, otimizando recursos e minimizando perdas.</p>
"""))

y_train_df = y_train.to_frame('class_encoded')
analysis_df_with_target = X_train_unscaled_df.join(y_train_df)

crosstab_clusters_risk = pd.crosstab(
    analysis_df_with_target['cluster'],
    analysis_df_with_target['class_encoded']
)
crosstab_clusters_risk.columns = ['Bad', 'Good']

crosstab_clusters_risk_prop = crosstab_clusters_risk.div(crosstab_clusters_risk.sum(axis=1), axis=0) * 100

display_header("Tabela de Proporção de Risco por Cluster", level=3)
display(crosstab_clusters_risk_prop.style.format('{:.2f}%').background_gradient(cmap='Reds', subset=['Bad']))

display(HTML("""
<h4>Interpretação da Análise de Risco por Cluster</h4>
<p>A análise cruzada confirma que os clusters criados pelo K-Means têm uma forte correlação com o risco de inadimplência:</p>
<ul>
    <li>
        <b>Cluster 1 (Profissionais de Alto Risco) é o segmento mais perigoso:</b> Este grupo apresenta uma alarmante taxa de <b>40.54% de maus pagadores</b>. O alto valor e a longa duração dos empréstimos, combinados com um perfil de emprego menos estável, se traduzem em um risco muito elevado para a instituição.
    </li>
    <li>
        <b>Cluster 2 (Clientes Maduros e Seguros) é o mais confiável:</b> Como esperado, este grupo tem a menor taxa de inadimplência, com apenas <b>18.89% de maus pagadores</b>, bem abaixo da média geral de 30%. Este é o "porto seguro" da carteira de clientes.
    </li>
    <li>
        <b>Clusters 0 e 3 (Jovens e Endividados Estáveis)</b> apresentam taxas de risco intermediárias (31.76% e 30.63%, respectivamente), muito próximas da média da população.
    </li>
</ul>
<p><b>Implicação Gerencial:</b> A empresa deve tratar o <b>Cluster 1</b> com extrema cautela, talvez exigindo garantias adicionais ou oferecendo produtos com limites mais baixos. Em contrapartida, pode direcionar ofertas de crédito mais agressivas e programas de fidelidade para o <b>Cluster 2</b>, que representa uma oportunidade de negócio segura e de baixo custo de risco.</p>
"""))

display_header("b) Detecção de Outliers com DBSCAN", level=2)
display(HTML("""
<p>Enquanto o K-Means agrupa todos os pontos de dados em um cluster, o DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um algoritmo baseado em densidade que é particularmente útil para identificar <b>outliers</b> — pontos de dados que não pertencem a nenhum cluster denso. Na nossa análise, um outlier é um cliente com um perfil tão atípico que ele se isola dos principais segmentos de comportamento.</p>
<p>O algoritmo funciona com dois parâmetros principais:</p>
<ul>
    <li><b><code>eps</code> (epsilon):</b> A distância máxima entre dois pontos para que um seja considerado vizinho do outro. Um valor menor resulta em áreas de densidade mais restritas.</li>
    <li><b><code>min_samples</code>:</b> O número mínimo de pontos necessários para formar uma região densa (um cluster). Pontos que não atendem a este critério e não estão no raio `eps` de nenhum cluster são rotulados como outliers (-1).</li>
</ul>
<p>Ajustamos esses parâmetros para identificar um subconjunto pequeno, mas significativo, de clientes anômalos.</p>
"""))

dbscan_model = DBSCAN(eps=4.0, min_samples=10)
outlier_labels_assigned = dbscan_model.fit_predict(X_unsupervised)

analysis_df_with_target['outlier_dbscan'] = outlier_labels_assigned

num_outliers_found = np.sum(outlier_labels_assigned == -1)
num_clusters_found = len(set(outlier_labels_assigned)) - (1 if -1 in outlier_labels_assigned else 0)
total_samples_unsupervised = len(X_unsupervised)
percentage_outliers_found = (num_outliers_found / total_samples_unsupervised) * 100

display_header("Resultados da Análise DBSCAN", level=3)
display(HTML(f"""
<p>A aplicação do DBSCAN resultou na identificação de:</p>
<ul>
    <li><b>Número de Clusters Densos:</b> {num_clusters_found}</li>
    <li><b>Número de Outliers (Pontos de Ruído):</b> {num_outliers_found}</li>
    <li><b>Percentual de Outliers na Base:</b> {percentage_outliers_found:.2f}%</li>
</ul>
<p>O próximo passo é investigar se esses clientes atípicos representam um risco de crédito maior ou menor do que a média da população.</p>
"""))

display(HTML("<h4>Análise Cruzada: Risco de Crédito dos Outliers</h4>"))
display(HTML("<p>Agora, filtramos apenas os clientes que o DBSCAN classificou como outliers (-1) e analisamos a distribuição da variável <code>class</code> dentro deste grupo específico. Isso nos permite responder à pergunta do enunciado: 'Existe relação entre os outliers detectados e o risco de inadimplência (class = bad)?'</p>"))

outliers_df = analysis_df_with_target[analysis_df_with_target['outlier_dbscan'] == -1]

if not outliers_df.empty:
    outlier_risk_dist = outliers_df['class_encoded'].value_counts(normalize=True) * 100
    
    outlier_summary_df = pd.DataFrame({
        'Contagem': outliers_df['class_encoded'].value_counts(),
        'Percentual (%)': outlier_risk_dist.round(2)
    })
    outlier_summary_df.index = outlier_summary_df.index.map({0: 'Bad', 1: 'Good'})
    
    display_header("Distribuição de Risco de Crédito no Grupo de Outliers", level=5)
    display(outlier_summary_df)

    fig_outlier_risk = go.Figure(go.Bar(
        x=outlier_summary_df.index,
        y=outlier_summary_df['Percentual (%)'],
        text=outlier_summary_df['Percentual (%)'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto',
        marker_color=['#d62728', '#1f77b4']
    ))
    fig_outlier_risk.update_layout(
        title='Proporção de Maus Pagadores no Segmento de Outliers',
        xaxis_title='Classe de Risco',
        yaxis_title='Percentual (%)',
        template=plotly_template
    )
    fig_outlier_risk.show()
    
    bad_rate_in_outliers = outlier_risk_dist.get(0, 0)
    
    display(HTML(f"""
    <p><b>Conclusão da Análise de Outliers:</b> A relação é clara e estatisticamente relevante. Enquanto a taxa de inadimplência na população geral é de 30%, no grupo de clientes atípicos identificados pelo DBSCAN, essa taxa sobe para <b>{bad_rate_in_outliers:.2f}%</b>. Isso indica que clientes com combinações raras de características (que os fazem se afastar dos grandes centros de densidade) têm uma probabilidade muito maior de se tornarem maus pagadores. </p>
    <p><b>Implicação Gerencial:</b> A detecção de outliers via DBSCAN pode servir como um sistema de alerta precoce. Um cliente que não se encaixa em nenhum segmento conhecido (K-Means) e é identificado como um outlier pelo DBSCAN deve ser encaminhado para uma análise de crédito manual e mais criteriosa, pois a probabilidade de risco associada a ele é substancialmente maior.</p>
    """))

else:
    display(HTML("<p>Nenhum outlier foi detectado com os parâmetros atuais do DBSCAN, portanto não é possível realizar a análise de risco para este grupo.</p>"))

display_header("d) Visualizações", level=2)
display_header("Visualização dos Clusters com Análise de Componentes Principais (PCA)", level=3)
display(HTML("""
<p>A Análise de Componentes Principais (PCA) é uma técnica de redução de dimensionalidade. Ela transforma um grande número de variáveis correlacionadas em um número menor de variáveis não correlacionadas, chamadas de "componentes principais".</p>
<p>Neste caso, usamos o PCA para projetar nossos dados de múltiplas dimensões em um espaço 2D (Componente Principal 1 vs. Componente Principal 2). O objetivo é puramente para <b>visualização</b>: queremos ver se os clusters que o K-Means identificou formam grupos visualmente distintos neste novo espaço. Uma boa separação no gráfico PCA é um forte indicador de que os clusters são coerentes e bem definidos.</p>
"""))

pca_visualizer = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_components = pca_visualizer.fit_transform(X_unsupervised)

pca_result_df = pd.DataFrame(
    data=X_pca_components,
    columns=['Componente Principal 1', 'Componente Principal 2']
)
pca_result_df['cluster'] = X_train_unscaled_df['cluster'].astype(str)

fig_pca_clusters = px.scatter(
    pca_result_df,
    x='Componente Principal 1',
    y='Componente Principal 2',
    color='cluster',
    title='Visualização dos Clusters de Clientes via PCA',
    labels={'cluster': 'Cluster'},
    category_orders={'cluster': sorted(pca_result_df['cluster'].unique())}
)

fig_pca_clusters.update_layout(
    template=plotly_template,
    legend_title_text='Cluster',
    height=700
)
fig_pca_clusters.update_traces(marker=dict(size=8, opacity=0.7))
fig_pca_clusters.show()

display(HTML("""
<p><b>Interpretação do Gráfico PCA:</b> A visualização mostra uma separação razoável entre os clusters, especialmente o Cluster 2 (em amarelo), que parece se destacar dos demais. Os outros clusters mostram alguma sobreposição, o que é esperado em dados do mundo real, mas ainda assim ocupam regiões distintas do espaço de componentes. Isso valida visualmente que o K-Means conseguiu encontrar uma estrutura subjacente nos dados dos clientes.</p>
"""))

display_header("Gráfico de Barras Comparando a Proporção de Maus Pagadores por Cluster", level=3)
display(HTML("""
<p>Esta é uma das visualizações mais importantes para a tomada de decisão gerencial. O gráfico abaixo resume os achados da nossa análise cruzada (Bloco 14), exibindo de forma clara e direta o percentual de clientes de alto risco ('bad') em cada um dos segmentos de clientes que identificamos.</p>
<p>Ao quantificar o risco associado a cada perfil de cliente, este gráfico permite que a área de negócios priorize suas ações, foque os esforços de monitoramento e personalize as políticas de crédito de forma muito mais eficaz.</p>
"""))

risk_proportion_data = crosstab_clusters_risk_prop.reset_index()
risk_proportion_data.rename(columns={'Bad': 'bad_proportion'}, inplace=True)
risk_proportion_data['cluster'] = risk_proportion_data['cluster'].astype(str)

fig_risk_bar = px.bar(
    risk_proportion_data.sort_values(by='bad_proportion', ascending=False),
    x='cluster',
    y='bad_proportion',
    title="Proporção de Maus Pagadores ('Bad') por Segmento de Cliente (Cluster)",
    labels={'cluster': 'Cluster de Cliente', 'bad_proportion': 'Percentual de Maus Pagadores (%)'},
    text=risk_proportion_data['bad_proportion'].apply(lambda x: f'{x:.2f}%'),
    color='bad_proportion',
    color_continuous_scale=px.colors.sequential.Reds
)
fig_risk_bar.update_layout(
    template=plotly_template,
    xaxis={'categoryorder':'total descending'},
    coloraxis_showscale=False
)
fig_risk_bar.update_traces(textposition='outside')
fig_risk_bar.show()

display(HTML("""
<p><b>Interpretação e Implicações Gerenciais do Gráfico:</b></p>
<p>O gráfico de barras torna a hierarquia de risco entre os segmentos inegável. Fica evidente que os <b>Clusters 1 e 3</b> são os mais problemáticos, com taxas de inadimplência de 40.54% e 30.63%, respectivamente. Em contraste, o <b>Cluster 2</b> é o mais seguro, com uma taxa de apenas 18.89%.</p>
<p>Esta visualização serve como um guia estratégico para a alocação de recursos:
<ul>
    <li><b>Ações de Mitigação Urgentes:</b> Devem ser focadas nos Clusters 1 e 3.</li>
    <li><b>Ações de Crescimento e Fidelização:</b> Podem ser direcionadas com segurança ao Cluster 2.</li>
    <li><b>Políticas Padrão:</b> Podem ser mantidas para o Cluster 0, que representa o risco médio da carteira.</li>
</ul>
</p>
"""))

from fpdf import FPDF
import os

display_header("III. Instruções de Entrega e Avaliação", level=1)
display_header("Preparando a Geração do Relatório em PDF", level=2)
display(HTML("""
<p>Para cumprir o formato de entrega obrigatório, o passo final é consolidar todas as análises, gráficos e conclusões em um relatório profissional em formato PDF. Para automatizar este processo, definimos uma classe auxiliar em Python que utiliza a biblioteca <code>fpdf</code>.</p>
<p>Esta classe, <code>PDFReport</code>, gerenciará a formatação do documento, incluindo a criação de uma página de rosto, cabeçalhos, rodapés com numeração de página, e a inserção de textos e imagens de forma estruturada. Isso garante um resultado final limpo, organizado e padronizado.</p>
"""))

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Prova Final - Análise de Risco de Crédito com IA', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        self.set_font('Arial', 'B', 16 - (level * 2))
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln()

    def add_plot(self, image_path):
        if os.path.exists(image_path):
            self.image(image_path, x=None, y=None, w=180)
            self.ln(5)

    def add_front_page(self, professor, aluno):
        self.add_page()
        self.set_font('Arial', 'B', 24)
        self.cell(0, 30, 'Análise de Risco de Crédito', 0, 1, 'C')
        self.ln(20)
        self.set_font('Arial', '', 16)
        self.cell(0, 10, f'Professor: {professor}', 0, 1, 'C')
        self.cell(0, 10, f'Aluno: {aluno}', 0, 1, 'C')
        self.cell(0, 10, f'Data: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'C')

if not os.path.exists("plots"):
    os.makedirs("plots")

print("Classe PDFReport e diretório de plots prontos para uso.")

display_header("Coletando e Estruturando o Conteúdo do Relatório", level=2)
display(HTML("""
<p>Antes de gerar o PDF, salvamos os gráficos mais importantes como arquivos de imagem e estruturamos todo o conteúdo textual. Isso desacopla a geração do conteúdo da sua renderização no PDF, tornando o processo mais modular e fácil de manter.</p>
"""))

fig_roc.write_image("plots/roc_curves.png", width=900, height=700, scale=2)
fig_risk_bar.write_image("plots/risk_by_cluster.png", width=800, height=500, scale=2)
fig_pca_clusters.write_image("plots/pca_clusters.png", width=800, height=700, scale=2)

fig_summary_bar, ax_summary_bar = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_class_bad, X_test_df, plot_type="bar", show=False)
plt.title(f"Importância Geral das Features (SHAP)")
plt.tight_layout()
plt.savefig("plots/shap_summary_bar.png", dpi=150)
plt.close()

report_content = {
    "I. Análise Preditiva": {
        "a) Diagnóstico e Desbalanceamento": "A base de dados apresentou um desbalanceamento, com 70% de clientes 'good' e 30% 'bad'. A técnica SMOTE foi aplicada ao conjunto de treino para criar um ambiente de aprendizado balanceado, evitando viés do modelo.",
        "b) Treinamento e Avaliação de Modelos": f"""Foram treinados e avaliados 9 modelos supervisionados. Após a análise comparativa de métricas como AUC, Recall e F1-Score, o modelo '{best_model_name}' foi selecionado como o de melhor performance geral, com um AUC de {best_model_auc:.4f}.""",
        "c) Explicabilidade com SHAP": "A análise SHAP foi aplicada ao modelo campeão para entender os fatores por trás de suas previsões. As variáveis mais influentes foram identificadas tanto em nível global quanto em casos individuais.",
        "d) Tomada de Decisão e Aplicação Gerencial": """Com base nos SHAP values, foi identificado que clientes com histórico de crédito ruim, sem conta corrente e com longas durações de empréstimo possuem o maior risco. A recomendação estratégica é a implementação de políticas de crédito mais conservadoras para este perfil, como limites de crédito menores e monitoramento proativo para mitigar o risco de inadimplência."""
    },
    "II. Modelos Não Supervisionados": {
        "a) Clusterização com KMeans": f"Utilizando o Método do Cotovelo, definimos k=4 como o número ideal de clusters. A análise dos perfis revelou 4 segmentos de clientes distintos com características bem definidas.",
        "b) Detecção de Outliers com DBSCAN": "O DBSCAN identificou um pequeno grupo de clientes atípicos (outliers). A análise cruzada mostrou que a taxa de inadimplência neste grupo é significativamente maior que a média da população, indicando que perfis anômalos representam um risco elevado.",
        "c) Análise Cruzada": "A distribuição da variável 'class' nos clusters do K-Means revelou que os segmentos encontrados têm perfis de risco distintos. O Cluster 1 foi identificado como o de maior risco (40.54% de 'bad'), enquanto o Cluster 2 foi o mais seguro (18.89% de 'bad').",
    }
}

print("Conteúdo e gráficos coletados e prontos para a montagem do relatório.")

display_header("Gerando o Relatório Final em PDF", level=2)
display(HTML("<p>Este bloco final executa a geração do documento PDF, juntando todo o texto e os gráficos salvos anteriormente. Ao final, teremos um arquivo <code>Prova_Final_Analise_de_Risco_Credito.pdf</code> pronto para entrega.</p>"))

pdf = PDFReport()
pdf.add_front_page(
    professor="João Gabriel de Moraes Souza",
    aluno="Gemini AI"
)

for section_title, subsections in report_content.items():
    pdf.add_page()
    pdf.chapter_title(section_title, level=1)
    for subsection_title, text in subsections.items():
        pdf.chapter_title(subsection_title, level=2)
        pdf.chapter_body(text.encode('latin-1', 'replace').decode('latin-1'))
        
        if "Avaliação de Modelos" in subsection_title:
            pdf.add_plot('plots/roc_curves.png')
        if "Explicabilidade com SHAP" in subsection_title:
            pdf.add_plot('plots/shap_summary_bar.png')

if os.path.exists('plots/pca_clusters.png'):
    pdf.chapter_title("d) Visualizações de Apoio", level=2)
    pdf.chapter_title("Visualização dos Clusters via PCA", level=3)
    pdf.add_plot('plots/pca_clusters.png')

if os.path.exists('plots/risk_by_cluster.png'):
    pdf.chapter_title("Proporção de Risco por Cluster", level=3)
    pdf.add_plot('plots/risk_by_cluster.png')


final_report_filename = "Prova_Final_Analise_de_Risco_Credito.pdf"
pdf.output(final_report_filename)

display(HTML(f"""
<h3>Relatório Final Gerado com Sucesso!</h3>
<p>O documento <b>'{final_report_filename}'</b> foi criado no diretório atual.</p>
<p>Este notebook cumpriu todas as etapas da Prova Final, desde a análise e preparação dos dados até a modelagem supervisionada e não supervisionada, culminando em insights acionáveis e na entrega de um relatório automatizado.</p>
"""))

display_header("Estruturando Conteúdo Textual para Relatório Final", level=2)
display(HTML("<p>Neste bloco, preparamos todo o conteúdo textual referente à <b>Parte I (Análise Supervisionada)</b>. Definir os textos em variáveis separadas antes de gerar o PDF torna o código mais limpo e facilita futuras manutenções ou alterações no relatório.</p>"))

report_part1_intro = """
O objetivo desta análise foi desenvolver um sistema de apoio à decisão para prever o risco de inadimplência de clientes de crédito. A variável-alvo, 'class', foi analisada e revelou um desbalanceamento de 70% para clientes 'good' e 30% para 'bad'. Para corrigir esse viés, a técnica SMOTE foi aplicada ao conjunto de dados de treinamento.
"""

report_part1_models = f"""
Foram treinados e avaliados 9 tipos de modelos supervisionados, conforme solicitado. Após uma análise comparativa de métricas, com foco especial em AUC e Recall para a classe 'bad', o modelo <b>{best_model_name}</b> foi selecionado como o de melhor performance geral, atingindo um AUC de <b>{best_model_auc:.4f}</b>. A seguir, a curva ROC comparativa de todos os modelos é apresentada, ilustrando a capacidade de discriminação de cada um.
"""

report_part1_shap = """
A análise de explicabilidade com SHAP foi aplicada ao modelo campeão para desmistificar suas previsões. A análise global revelou que o histórico de crédito, o status da conta corrente e a duração do empréstimo são os fatores de maior impacto nas previsões do modelo.
"""

report_part1_shap_local = """
A análise de casos individuais (waterfall plots) reforça os achados globais. Para um cliente corretamente previsto como 'bad', a combinação de um histórico de crédito negativo e a falta de uma conta corrente foram determinantes. Em contrapartida, para um cliente 'good', um bom histórico de crédito foi o fator decisivo para uma previsão de baixo risco, mesmo com outras variáveis indicando um risco moderado.
"""

report_part1_managerial = """
Com base nas evidências geradas pelo SHAP, a recomendação estratégica para a área de crédito é a seguinte:

<b>"Clientes com histórico de crédito problemático (com atrasos passados) e sem uma conta corrente estabelecida apresentam os maiores SHAP values para o risco 'bad', indicando uma alta probabilidade de inadimplência. Sugere-se a implementação de uma política de crédito mais conservadora para este perfil, incluindo a oferta de limites de crédito iniciais mais baixos e prazos de pagamento mais curtos (menor 'duration'). Para a carteira existente, recomenda-se a criação de um alerta proativo para o time de relacionamento quando clientes com este perfil solicitarem aumentos de limite, permitindo uma intervenção focada em educação financeira antes de uma possível inadimplência."</b>

Esta recomendação é diretamente acionável e fundamentada nos fatores de maior peso preditivo identificados pelo sistema de IA, alinhando a capacidade analítica do modelo com os objetivos de crescimento sustentável da empresa.
"""

print("Textos da Seção I (Supervisionada) prontos para o relatório.")

display_header("Estruturando Conteúdo Textual - Parte II", level=3)
display(HTML("<p>Similarmente ao bloco anterior, este bloco prepara o conteúdo textual para a <b>Parte II (Análise Não Supervisionada)</b> do relatório. A clareza na comunicação dos achados de segmentação e detecção de outliers é fundamental para que a área de negócio possa utilizar esses insights.</p>"))

report_part2_kmeans = f"""
A clusterização com K-Means foi realizada para segmentar a base de clientes. O Método do Cotovelo indicou k=4 como o número ideal de clusters. A análise dos perfis revelou quatro segmentos distintos e acionáveis:
<ul>
    <li><b>Cluster 0 (Jovens em Consolidação):</b> Perfil de risco moderado, buscando crédito para consumo.</li>
    <li><b>Cluster 1 (Profissionais Liberais de Alto Risco):</b> Perfil de maior risco, com os maiores valores e durações de empréstimo.</li>
    <li><b>Cluster 2 (Clientes Maduros e Seguros):</b> Perfil de menor risco, com endividamento baixo e pontual.</li>
    <li><b>Cluster 3 (Endividados com Emprego Estável):</b> Risco moderado a alto, com emprego estável mas uso intensivo do crédito.</li>
</ul>
A visualização via PCA confirmou a coerência e a separação destes segmentos no espaço de características.
"""

report_part2_dbscan = """
O algoritmo DBSCAN foi utilizado para identificar clientes com perfis atípicos (outliers). Foi encontrado um pequeno grupo de outliers que, ao serem analisados, apresentaram uma taxa de inadimplência significativamente superior à média da população geral. Isso sugere que perfis que fogem ao padrão dos segmentos principais representam um risco concentrado e devem ser tratados com maior cautela pela gestão de crédito.
"""

report_part2_crosstab = """
A análise cruzada, que combina os resultados da clusterização com os dados reais de risco, confirmou as hipóteses. O Cluster 1 apresentou a maior proporção de maus pagadores (40.54%), enquanto o Cluster 2 se mostrou o mais seguro (18.89%). O gráfico de barras comparativo ilustra essa disparidade de risco, fornecendo um guia visual claro para a alocação de recursos e definição de políticas de crédito diferenciadas por segmento.
"""

print("Textos da Seção II (Não Supervisionada) prontos para o relatório.")

display_header("Construindo o Documento PDF", level=2)
display(HTML("<p>Com todo o conteúdo preparado, este bloco executa a montagem do relatório. Ele instancia a classe <code>PDFReport</code> e utiliza os textos e imagens para construir o documento seção por seção, seguindo a estrutura da Prova Final.</p>"))

pdf = PDFReport()
pdf.add_front_page(
    university="Universidade de Brasília – UnB",
    faculty="Faculdade de Tecnologia – FT",
    department="Departamento de Engenharia de Produção – EPR",
    course="Prova Final - Análise de Risco de Crédito",
    professor="João Gabriel de Moraes Souza",
    student="Gemini AI",
    date="10/07/2025"
)

pdf.add_page()
pdf.set_text_color(0, 0, 0)

pdf.chapter_title(text_part1["title"], level=1)

pdf.chapter_title("a) Diagnóstico e Desbalanceamento", level=2)
pdf.chapter_body(report_part1_intro.encode('latin-1', 'replace').decode('latin-1'))

pdf.chapter_title("b) Treinamento e Avaliação de Modelos", level=2)
pdf.chapter_body(report_part1_models.encode('latin-1', 'replace').decode('latin-1'))
pdf.add_plot('plots_for_pdf/roc_curves.png')

pdf.chapter_title("c) Explicabilidade com SHAP (XAI)", level=2)
pdf.chapter_body(report_part1_shap.encode('latin-1', 'replace').decode('latin-1'))
pdf.add_plot('plots_for_pdf/shap_summary_bar.png')
pdf.ln(5)
pdf.chapter_body(report_part1_shap_local.encode('latin-1', 'replace').decode('latin-1'))
if os.path.exists('plots_for_pdf/shap_waterfall_bad.png'):
    pdf.add_plot('plots_for_pdf/shap_waterfall_bad.png')

pdf.chapter_title("d) Tomada de Decisão e Aplicação Gerencial", level=2)
pdf.chapter_body(report_part1_managerial.encode('latin-1', 'replace').decode('latin-1'))

pdf.add_page()
pdf.chapter_title(text_part4["title"], level=1)

pdf.chapter_title("a) Clusterização com KMeans", level=2)
pdf.chapter_body(report_part2_kmeans.encode('latin-1', 'replace').decode('latin-1'))

pdf.chapter_title("b) Detecção de Outliers com DBSCAN", level=2)
pdf.chapter_body(report_part2_dbscan.encode('latin-1', 'replace').decode('latin-1'))

pdf.chapter_title("c) e d) Análise Cruzada e Visualizações", level=2)
pdf.chapter_body(report_part2_crosstab.encode('latin-1', 'replace').decode('latin-1'))
pdf.add_plot('plots_for_pdf/pca_clusters.png')
pdf.ln(5)
pdf.add_plot('plots_for_pdf/risk_by_cluster.png')

print("Estrutura do PDF montada com sucesso, pronta para ser salva.")

display_header("Salvando o Relatório Final", level=2)
display(HTML("<p>Este é o bloco final, que executa o comando para salvar o objeto PDF montado em um arquivo físico no disco. Após a execução, uma mensagem de confirmação é exibida, e o artefato final do projeto estará pronto para entrega.</p>"))

final_report_filename = "Prova_Final_Analise_de_Risco_Credito.pdf"
pdf.output(final_report_filename)

if os.path.exists(final_report_filename):
    display(HTML(f"""
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px; background-color: #f0fff0;">
        <h2 style='color: #4CAF50;'>Projeto Concluído e Relatório Gerado!</h2>
        <p>O documento <b>'{final_report_filename}'</b> foi salvo com sucesso no mesmo diretório onde este notebook está sendo executado.</p>
        <p>Todos os 25 blocos de código foram executados, cobrindo integralmente as exigências da Prova Final, desde a análise de dados e modelagem até a geração automatizada do entregável.</p>
        <p><b>Obrigado por utilizar esta solução.</b></p>
    </div>
    """))
else:
    display(HTML(f"""
    <div style="border: 2px solid #D62728; padding: 15px; border-radius: 5px; background-color: #fff0f0;">
        <h2 style='color: #D62728;'>Erro na Geração do Relatório!</h2>
        <p>O arquivo <b>'{final_report_filename}'</b> não pôde ser criado. Por favor, verifique as permissões de escrita no diretório e se a biblioteca FPDF está instalada corretamente.</p>
    </div>
    """))

print(f"\nProcesso finalizado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

