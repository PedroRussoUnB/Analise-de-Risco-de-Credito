from IPython.display import display, HTML
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow import keras
from tensorflow.keras import layers

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

display(HTML("<h1>Projeto Final: Análise de Risco de Crédito com Machine Learning</h1>"))
display(HTML("""
<p>Este notebook apresenta uma solução completa para o desafio de análise de risco de crédito, utilizando técnicas de aprendizado de máquina supervisionado e não supervisionado. O objetivo é construir um sistema de apoio à decisão que seja não apenas preditivo, mas também explicável e acionável.</p>
<h3>1. Inicialização do Ambiente</h3>
<p>A primeira célula é dedicada à importação de todas as bibliotecas necessárias para o projeto e à configuração de parâmetros globais para garantir a reprodutibilidade e a consistência visual da análise.</p>
<ul>
    <li><b>Manipulação de Dados:</b> <code>pandas</code> e <code>numpy</code>.</li>
    <li><b>Visualização:</b> <code>matplotlib</code>, <code>seaborn</code>, e <code>plotly</code> para gráficos estáticos e interativos.</li>
    <li><b>Pré-processamento e Modelagem:</b> <code>scikit-learn</code> (<code>sklearn</code>) para divisão de dados, pré-processamento, e a maioria dos algoritmos de ML.</li>
    <li><b>Balanceamento de Dados:</b> <code>imbalanced-learn</code> (<code>imblearn</code>) para a técnica SMOTE.</li>
    <li><b>Modelos Avançados:</b> <code>xgboost</code> e <code>lightgbm</code> para algoritmos de boosting de alta performance.</li>
    <li><b>Redes Neurais:</b> <code>tensorflow.keras</code> para a construção do modelo de rede neural.</li>
    <li><b>Explicabilidade:</b> <code>shap</code> para interpretar as previsões dos modelos.</li>
    <li><b>Utilitários:</b> <code>warnings</code> para gerenciar avisos, <code>os</code> e <code>datetime</code> para manipulação de arquivos e datas.</li>
</ul>
"""))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

print("Bibliotecas importadas e ambiente configurado com sucesso.")