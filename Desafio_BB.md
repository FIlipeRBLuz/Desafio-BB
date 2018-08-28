---


---

<h1 id="desafio-bb---ml-classificação">Desafio BB - ML Classificação</h1>
<p>O objetivo do projeto é classificar de forma mais assertiva possível pessoas com salário a cima de 50K. Esta classificação deverá ser feita via Machine Learning, utilizando as mais diversas técnicas e validando o resultado com medida de acurácia e log loss.</p>
<h1 id="files">Files</h1>
<p>Foi utilizado como data set registros de cidadães Americanos obtido via censo.</p>
<h2 id="importações-de-libraries">Importações de libraries</h2>
<p>import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
from sklearn.pipeline import Pipeline<br>
from sklearn.svm import SVC<br>
from sklearn.model_selection import train_test_split, GridSearchCV<br>
from sklearn.metrics import  classification_report,confusion_matrix<br>
from sklearn.linear_model import LogisticRegression</p>
<h2 id="importando-dataset-do-projeto">Importando dataset do projeto</h2>
<p>df = pd.read_csv(‘census.csv’)</p>
<h2 id="verificando-registros-e-entendendo-sua-distribuição">Verificando registros e entendendo sua distribuição</h2>
<p><a href="http://df.info">df.info</a>()<br>
df.head()<br>
df.tail()<br>
df.shape<br>
df.columns<br>
df.describe()</p>
<p>RangeIndex: 5533 entries, 0 to 5532<br>
Data columns (total 14 columns):<br>
age                5533 non-null int64<br>
workclass          5533 non-null object<br>
education_level    5533 non-null object<br>
education-num      5533 non-null float64<br>
marital-status     5533 non-null object<br>
occupation         5533 non-null object<br>
relationship       5533 non-null object<br>
race               5533 non-null object<br>
sex                5533 non-null object<br>
capital-gain       5532 non-null float64<br>
capital-loss       5532 non-null float64<br>
hours-per-week     5532 non-null float64<br>
native-country     5532 non-null object<br>
income             5532 non-null object<br>
dtypes: float64(4), int64(1), object(9)<br>
memory usage: 605.2+ KB</p>
<p>age     education-num  capital-gain  capital-loss  hours-per-week<br>
count  5533.000000    5533.000000   5532.000000   5532.000000     5532.000000<br>
mean     38.454003      10.114947   1065.107918     92.788322       41.074295<br>
std      13.098876         2.535321   7251.498867    409.629931       11.623411<br>
min      17.000000       1.000000      0.000000      0.000000        1.000000<br>
25%      28.000000       9.000000      0.000000      0.000000       40.000000<br>
50%      37.000000      10.000000      0.000000      0.000000       40.000000<br>
75%      47.000000      13.000000      0.000000      0.000000       45.000000<br>
max      90.000000      16.000000  99999.000000   2824.000000       99.000000</p>
<h2 id="verificação-de-registros-nan">Verificação de Registros NaN</h2>
<p>Tem como objetivo analisar a qualidade do dataset, para executar modelos de<br>
Machine Learning.</p>
<p>print(‘Quantidade de registros nulos em cada coluna do dataset:’)<br>
print(df.isnull().sum())</p>
<p>Quantidade de registros nulos em cada coluna do dataset:<br>
age                0<br>
workclass          0<br>
education_level    0<br>
education-num      0<br>
marital-status     0<br>
occupation         0<br>
relationship       0<br>
race               0<br>
sex                0<br>
capital-gain       1<br>
capital-loss       1<br>
hours-per-week     1<br>
native-country     1<br>
income             1<br>
dtype: int64</p>
<h2 id="exclusão-da-única-linha-nula">Exclusão da única linha Nula</h2>
<p>Limpeza de NaN, existe uma linha de registro sem informações importantes bem como informação categorica. O que gera uma dificuldade de aplicar uma categoriza e enviesar o modelo.</p>
<p>df = df.dropna()</p>
<h1 id="verificando-se-registro-foi-excluído">Verificando se registro foi excluído</h1>
<p>Verificação de NaN values apos excluir linhas com NaN.</p>
<p>print(‘Quantidade de registros nulos em cada coluna do dataset, após limpeza:’)<br>
print(df.isnull().sum())<br>
<a href="http://df.info">df.info</a>()</p>
<p>Quantidade de registros nulos em cada coluna do dataset, após limpeza:<br>
age                0<br>
workclass          0<br>
education_level    0<br>
education-num      0<br>
marital-status     0<br>
occupation         0<br>
relationship       0<br>
race               0<br>
sex                0<br>
capital-gain       0<br>
capital-loss       0<br>
hours-per-week     0<br>
native-country     0<br>
income             0<br>
dtype: int64</p>
<p>&lt;class ‘pandas.core.frame.DataFrame’&gt;<br>
Int64Index: 5532 entries, 0 to 5531<br>
Data columns (total 14 columns):<br>
age                5532 non-null int64<br>
workclass          5532 non-null object<br>
education_level    5532 non-null object<br>
education-num      5532 non-null float64<br>
marital-status     5532 non-null object<br>
occupation         5532 non-null object<br>
relationship       5532 non-null object<br>
race               5532 non-null object<br>
sex                5532 non-null object<br>
capital-gain       5532 non-null float64<br>
capital-loss       5532 non-null float64<br>
hours-per-week     5532 non-null float64<br>
native-country     5532 non-null object<br>
income             5532 non-null object<br>
dtypes: float64(4), int64(1), object(9)<br>
memory usage: 648.3+ KB</p>
<h1 id="funções-criadas-para-reuso">Funções Criadas para reuso</h1>
<p>def heat_corr (data):<br>
‘’‘Analisando a correlação entre as features<br>
do dataset<br>
‘’’<br>
sns.heatmap(data.corr(),square=True,cmap=‘RdYlGn’)<br>
plt.title(‘Correlação entre as features’)<br>
plt.show()</p>
<p>def distribuicao (data):<br>
‘’’<br>
Esta função exibirá a quantidade de registros únicos para cada coluna<br>
existente no dataset</p>
<pre><code>dataframe -&gt; Histogram
'''
# Calculando valores unicos para cada label: num_unique_labels
num_unique_labels = data.apply(pd.Series.nunique)

# plotando valores
num_unique_labels.plot( kind='bar')

# Nomeando os eixos
plt.xlabel('Campos')
plt.ylabel('Número de Registros únicos')
plt.title('Distribuição dos dados do DataSet')

# Exibindo gráfico
plt.show()
</code></pre>
<p>def compute_log_loss(predicted, actual, eps=1e-14):<br>
‘’’<br>
Computa a medida de avaliação sobre perda (log loss)<br>
Utilizada para avaliação do modelo<br>
‘’’</p>
<pre><code>predicted = np.clip( predicted , eps , 1- eps)  
loss = -1 * np.mean(  actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

return loss
</code></pre>
<p>def ac_desemp (cm):<br>
ac = 0<br>
ac = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[1,0] + cm[0,1])</p>
<pre><code>print("Acuracia do modelo é : " + str(ac))
</code></pre>
<h1 id="análise-eda">Análise EDA</h1>
<ul>
<li>O objetivo é analisar de forma gráfica o data set a fim de encontrar inconsistências, outliers e entender a distribuição dos dados no data set.</li>
</ul>
<p><a href="https://drive.google.com/file/d/1Cal4zi1k1FBQObf-c7w8p7tZfDUqlZq6/view?usp=sharing">heatmap correlação das colunas</a></p>
<h2 id="criação-de-nova-coluna">Criação de nova coluna</h2>
<p>Criando Coluna para dizer se salário está a cima de 50k ou Não.<br>
Este processo torna-se necessário uma vez que o problema é classificar<br>
de forma acertiva pessoas com salário a cima de 50. Algotirimos de classificação<br>
trabalham melhor com features numéricas, apesar de não ser uma regra obrigatória.</p>
<p>df[‘col_pred’] = np.where(df.income == ‘&gt;50K’, 1,0)</p>
<h2 id="análise-percentual-50k-e-50k">Análise percentual &gt;50k e &lt;50K</h2>
<p>recs = df[‘age’].count()<br>
qt_maior = (df.col_pred == 1).sum()<br>
perc = round(float((qt_maior * 100) / recs),2)<br>
print(‘Percentual de pessoas com salário acima de 50K: ’ + str(perc) + ’ %’)<br>
print(‘Percentual de pessoas com salário igual ou a baixo de 50K: ’ + str(100 - perc) + ’ %’)</p>
<p><strong>output:</strong><br>
Percentual de pessoas com salário acima de 50K: 25.18 %<br>
Percentual de pessoas com salário igual ou a baixo de 50K: 74.82 %</p>
<h2 id="analisando-correlação-após-criar-nova-coluna">Analisando Correlação Após criar nova Coluna</h2>
<p>Analisando correlação das features após a criação da coluna col_pred que terá a função de Y no nosso modelo de Classificação.</p>
<p><img src="https://drive.google.com/file/d/1r3si58dLYNNX7ST8Ra8tox9QDD2soLxS/view?usp=sharing" alt="Heat Map Correlação"></p>
<h2 id="analisando-distribuição-dos-dados">Analisando Distribuição dos dados</h2>
<p>Objetivo é entender a distribuição de dados em cada coluna no data set do projeto.<br>
Para identificar possíveis ações de preparo no data set.</p>
<p>distribuicao(df)</p>
<p><img src="https://drive.google.com/file/d/1MXupLAFB4nwtkCZK-mJIB8_rNdRzbHQE/view?usp=sharing" alt="enter image description here"></p>
<h2 id="split-do-dataset">Split do DataSet</h2>
<p>Uma vez tratado, eliminando inconsistências encontradas, partiremos para a criação do modelo de classificação. O processo de split é feito para podermos utilizar para ensinar a máquina com parte do data set (Train) e a outra parte (Teste) será utilizada para validação da assertividade do modelo, ou seja, o quão bem a maquina conseguirá classificar dados os quais ela nunca viu.</p>
<p>#Excluindo coluna income e col_pred, para gerar X<br>
X = df.drop([‘col_pred’,‘income’],axis=1)</p>
<p>#Separando coluna col_pred, a qual será nossa Y<br>
y = df[‘col_pred’]</p>
<p>No passo a baixo, estou separando colunas numéricas das não numéricas<br>
com o intuito de posteriormente poder avaliar os modelos apenas com colunas numéricas<br>
e para poder realizar processos de tratamento nas features de texto.</p>
<p>#Buscando colunas descritivas do data set<br>
LABELS = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] in [‘object’]]<br>
NUMB = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] not in [‘object’]]</p>
<h1 id="aplicação-de-técnica-dumy">Aplicação de Técnica Dumy</h1>
<p>Foi escolhida a tecnica de Dummy columns, com exclusão de uma coluna para cada<br>
grupo de campo descritivo, seguindo melhores práticas.<br>
O método dumy gera colunas para cada campo descritivo do data set, por Ex.: Estado: DF,SP,RJ. Será criada coluna Estado_DF,Estado_RJ.<br>
Quando o registro for DF a coluna Estado_DF terá valor 1, Estado_RJ terá valor 0. Quando Estado_df e Estado_rj forem respectivamente 0, significa que o Estado é SP.<br>
Este processo é feito para melhorar a classificação dos algoritimos que usam medida euclidiana. Foi escolhido no lugar do Hot Encoder pois facilita o processo de ajuste do dataset apesar de aumentar significativamente a quantidade de colunas.</p>
<p>#Aplicando tecnica dummy nas colunas descritivas<br>
X_dum = pd.get_dummies(X[LABELS], drop_first=True)</p>
<h2 id="aplicação-de-modelo-de-classificação">Aplicação de Modelo de Classificação</h2>
<p>Após os ajustes no dataset iremos aplicar os modelos de ML pra classificar os registros do dataset, ensinando assim a máquina a classificar de forma mais assertiva possível novos registros, posteriormente ao aprendizado.</p>
<h2 id="utilizarei-inicialmente-todas-colunas-do-data-set">Utilizarei inicialmente todas colunas do data set</h2>
<h3 id="svm">SVM</h3>
<h4 id="setup-the-pipeline-steps-steps">Setup the pipeline steps: steps</h4>
<p>steps = [(‘SVM’, SVC())]</p>
<h4 id="create-the-pipeline-pipeline">Create the pipeline: pipeline</h4>
<p>pipeline = Pipeline(steps)</p>
<h4 id="create-training-and-test-sets">Create training and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X_dum,y, test_size=0.25, random_state=0)</p>
<h4 id="fit-the-pipeline-to-the-train-set">Fit the pipeline to the train set</h4>
<p>pipeline.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set">Predict the labels of the test set</h4>
<p>y_pred = pipeline.predict(X_test)</p>
<h4 id="compute-metrics">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.87      0.92      0.89      1069
      1       0.66      0.52      0.58       314
</code></pre>
<p>avg / total       0.82      0.83      0.82      1383</p>
<p>Log Loss Result: 5.500945996700275<br>
Acuracia do modelo é : 0.8293564714389009</p>
<h3 id="logisticregression">LogisticRegression</h3>
<p>Para tentar melhorar o desempenho do SVM utilizei o método de tuning<br>
Grid Search, o qual verifica o desempenho do algoritimo com diferentes parametos selecionados retornando os parâmetros que obtiveram o melhor desempenho.<br>
Utilizarei as principais opções de tuning do algoritimo, levando em consideração o processo de regularização dos dados, como opção l1 = Lasso e l2 = Ridge.<br>
Acredito que a utilização de regularização nesse modelo faz dele candidato a modelo ideal para o problema.</p>
<h4 id="criação-da-grade-de-hiperparametros">Criação da grade de hiperparametros</h4>
<p>c_space = np.logspace(-5, 8, 45)<br>
param_grid = {‘C’: c_space, ‘penalty’: [‘l1’, ‘l2’]}</p>
<h4 id="instantiate-the-logistic-regression-classifier-logreg">Instantiate the logistic regression classifier: logreg</h4>
<p>logreg = LogisticRegression()</p>
<h4 id="create-train-and-test-sets">Create train and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X_dum,y,test_size=0.25,random_state = 0)</p>
<h4 id="instantiate-the-gridsearchcv-object-logreg_cv">Instantiate the GridSearchCV object: logreg_cv</h4>
<p>logreg_cv = GridSearchCV(logreg,param_grid,cv=5)</p>
<h4 id="fit-it-to-the-training-data">Fit it to the training data</h4>
<p>logreg_cv.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-1">Predict the labels of the test set</h4>
<p>y_pred = logreg_cv.predict(X_test)</p>
<h4 id="print-the-optimal-parameters-and-best-score">Print the optimal parameters and best score</h4>
<p>print(“Tuned Logistic Regression Parameter: {}”.format(logreg_cv.best_params_))<br>
print(“Tuned Logistic Regression Accuracy: {}”.format(logreg_cv.best_score_))</p>
<h4 id="compute-metrics-1">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<p>Tuned Logistic Regression Parameter: {‘C’: 2.0805675382171716, ‘penalty’: ‘l1’}<br>
Tuned Logistic Regression Accuracy: 0.8201976379850566<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.87      0.91      0.89      1069
      1       0.63      0.55      0.59       314
</code></pre>
<p>avg / total       0.82      0.83      0.82      1383</p>
<p>Log Loss Result: 5.640809149966951</p>
<h3 id="random-forest-classification">Random Forest Classification</h3>
<p>Utilizarei o Random Forest por ser um algritimo com alto potêncial de classificação<br>
apesar de ter o risco de muitos falsos positivos é um algorítimo muito utilizado pela comunidade, obtendo resultados significativos em competições de Data Science.<br>
Por ser uma junção de Decision Trees pode ser que consigamos um melhor desempenho na classificação do dataset utilizado.</p>
<h4 id="splitting-the-dataset-into-the-training-set-and-test-set">Splitting the dataset into the Training set and Test set</h4>
<p>from sklearn.cross_validation import train_test_split<br>
X_train, X_test, y_train, y_test = train_test_split(X_dum, y, test_size = 0.25,random_state = 0)</p>
<h4 id="feature-scaling">Feature Scaling</h4>
<p>Aplicando método de ajuste da escala para aproximar os valores e evitar grande<br>
disparidade entre as similaridades dos dados.</p>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="fitting-random-forest-classification-to-the-training-set">Fitting Random Forest Classification to the Training set</h4>
<p>from sklearn.ensemble import RandomForestClassifier<br>
classifier = RandomForestClassifier(n_estimators = 10, criterion = ‘entropy’, random_state = 0)<br>
classifier.fit(X_train, y_train)</p>
<h4 id="predicting-the-test-set-results">Predicting the Test set results</h4>
<p>y_pred = classifier.predict(X_test)</p>
<h4 id="making-the-confusion-matrix">Making the Confusion Matrix</h4>
<p>cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<h4 id="compute-metrics-2">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<p>Acuracia do modelo é : 0.8221258134490239<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.87      0.91      0.89      1069
      1       0.63      0.54      0.58       314
</code></pre>
<p>avg / total       0.81      0.82      0.82      1383</p>
<p>Log Loss Result: 5.734044699647114</p>
<h3 id="k-nn">K-NN</h3>
<p>O K-NN é recomendado para modelos de classificação, utiliza similaridade dos dados<br>
para promover a classificação desejada.</p>
<h4 id="splitting-the-dataset-into-the-training-set-and-test-set-1">Splitting the dataset into the Training set and Test set</h4>
<p>from sklearn.cross_validation import train_test_split<br>
X_train, X_test, y_train, y_test = train_test_split(X_dum, y, test_size = 0.25, random_state = 0)</p>
<h4 id="feature-scaling-1">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc_X = StandardScaler()<br>
X_train = sc_X.fit_transform(X_train)<br>
X_test = sc_X.transform(X_test)</p>
<h4 id="fitting-knn-classifier-to-the-training-set">Fitting KNN Classifier to the Training set</h4>
<p>from sklearn.neighbors import KNeighborsClassifier<br>
classifier = KNeighborsClassifier( n_neighbors = 5,<br>
metric = ‘minkowski’,<br>
p = 2 )<br>
classifier.fit(X_train, y_train)</p>
<p>####Predict the Test set Results<br>
y_pred = classifier.predict(X_test)</p>
<h4 id="making-the-confusion-matrix-1">Making the Confusion Matrix</h4>
<p>cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<h4 id="compute-metrics-3">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<p>Acuracia do modelo é : 0.8170643528561099<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.87      0.90      0.88      1069
      1       0.61      0.54      0.57       314
</code></pre>
<p>avg / total       0.81      0.82      0.81      1383</p>
<p>Log Loss Result: 5.89721038055673</p>
<h1 id="teste-dos-modelos-com-todas-colunas-numéricas">Teste dos Modelos com Todas colunas numéricas</h1>
<p>Após executar os modelos com o data set completo e identificar que os mesmos estavam<br>
estagnando em relação a acurácia e log loss alto, resolvi testar os algorítimos utilizando<br>
apenas as features numéricas com maior correlação.<br>
Usei como Base o heat map gerado nos passos iniciais para selecionar as features que seriam utilizadas nos testes seguintes.</p>
<h3 id="separando-dataset-com-colunas-numéricas">Separando dataset com colunas numéricas</h3>
<p>X_num = X[NUMB]</p>
<h4 id="verificando-a-escala-das-colunas-numéricas">Verificando a escala das colunas numéricas</h4>
<p>print(“Média das medidas sem escala: {}”.format(np.mean(X_num)))<br>
print(“Desvio Padrão das medidas sem escala: {}”.format(np.std(X_num)))</p>
<p>Aplicando método scalling para aproximar os valores das colunas numéricas<br>
a fim de tentar melhorar o desempenho das classificações uma vez que medidas muito<br>
espaçadas inferem no resultado do algorítimo.</p>
<p>from sklearn.preprocessing import scale</p>
<h4 id="scalling">Scalling</h4>
<p>X_scaled = scale(X_num)</p>
<p>Média das medidas sem escala: age                 38.455712<br>
education-num       10.115148<br>
capital-gain      1065.107918<br>
capital-loss        92.788322<br>
hours-per-week      41.074295<br>
dtype: float64<br>
Desvio Padrão das medidas sem escala: age                 13.098259<br>
education-num        2.535277<br>
capital-gain      7250.843424<br>
capital-loss       409.592905<br>
hours-per-week      11.622360<br>
dtype: float64</p>
<h3 id="print-the-mean-and-standard-deviation-of-the-scaled-features">Print the mean and standard deviation of the scaled features</h3>
<p>print(“Média das medidas com escala: {}”.format(np.mean(X_scaled)))<br>
print(“Desvio Padrão das medidas com escala: {}”.format(np.std(X_scaled)))</p>
<p>Média das medidas com escala: 1.1328609778387714e-16<br>
Desvio Padrão das medidas com escala: 1.0</p>
<h3 id="svm-1">SVM</h3>
<p>SVM com colunas numéricas e ajustadas via scalling, objetivo é baixar o log loss mantendo a acurácia do modelo de classificação</p>
<h4 id="setup-the-pipeline-steps-steps-1">Setup the pipeline steps: steps</h4>
<p>steps = [(‘SVM’, SVC(probability=True))]</p>
<h4 id="create-the-pipeline-pipeline-1">Create the pipeline: pipeline</h4>
<p>pipeline = Pipeline(steps)</p>
<h4 id="create-training-and-test-sets-1">Create training and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.25, random_state=30)</p>
<h4 id="fit-the-pipeline-to-the-train-set-1">Fit the pipeline to the train set</h4>
<p>pipeline.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-2">Predict the labels of the test set</h4>
<p>y_pred = pipeline.predict(X_test)</p>
<h4 id="compute-metrics-4">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))<br>
cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<p>precision    recall  f1-score   support</p>
<pre><code>      0       0.83      0.97      0.89      1033
      1       0.84      0.41      0.55       350
</code></pre>
<p>avg / total       0.83      0.83      0.81      1383</p>
<p>Log Loss Result: 5.50091361965321<br>
Acuracia do modelo é : 0.8293564714389009</p>
<p>O resultado apresentado pelo SVM com escala foi satisfatório em relação<br>
a melhora da acurácia do modelo, porém a tentativa de baixar o log loss não se confirmou<br>
neste modelo, talvez seja necessário trabalhar com probabilidade ao realizar a predição.</p>
<h3 id="logisticregression-1">LogisticRegression</h3>
<h4 id="create-the-hyperparameter-grid">Create the hyperparameter grid</h4>
<p>c_space = np.logspace(-15, 8, 15)<br>
param_grid = {‘C’: c_space, ‘penalty’: [‘l1’, ‘l2’]}</p>
<h4 id="instantiate-the-logistic-regression-classifier-logreg-1">Instantiate the logistic regression classifier: logreg</h4>
<p>logreg = LogisticRegression()</p>
<h4 id="create-train-and-test-sets-1">Create train and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.25,random_state = 30)</p>
<p>‘’‘Na tentativa de optimizar o processo de tuning, aumentei a quantidade de verificações<br>
Passanco cv de 5 para 10 tentativas’’’</p>
<p>logreg_cv = GridSearchCV(logreg,param_grid,cv=10)</p>
<h4 id="fit-it-to-the-training-data-1">Fit it to the training data</h4>
<p>logreg_cv.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-3">Predict the labels of the test set</h4>
<p>y_pred = logreg_cv.predict(X_test)</p>
<h4 id="print-the-optimal-parameters-and-best-score-1">Print the optimal parameters and best score</h4>
<p>print(“Tuned Logistic Regression Parameter: {}”.format(logreg_cv.best_params_))<br>
print(“Tuned Logistic Regression Accuracy: {}”.format(logreg_cv.best_score_))</p>
<h4 id="compute-metrics-5">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_test,y_pred)))<br>
cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<p>Tuned Logistic Regression Parameter: {‘C’: 26.826957952797162, ‘penalty’: ‘l1’}<br>
Tuned Logistic Regression Accuracy: 0.7980236201494336<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.83      0.94      0.88      1033
      1       0.72      0.44      0.55       350
</code></pre>
<p>avg / total       0.80      0.81      0.80      1383</p>
<p>Log Loss Result: 5.990497386615262<br>
Acuracia do modelo é : 0.8141720896601591</p>
<h3 id="random-forest-classification-1">Random Forest Classification</h3>
<h4 id="splitting-the-dataset-into-the-training-set-and-test-set-2">Splitting the dataset into the Training set and Test set</h4>
<p>from sklearn.cross_validation import train_test_split<br>
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.35, random_state = 0)</p>
<h4 id="feature-scaling-2">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="fitting-random-forest-classification-to-the-training-set-1">Fitting Random Forest Classification to the Training set</h4>
<p>from sklearn.ensemble import RandomForestClassifier<br>
classifier = RandomForestClassifier(n_estimators = 10, criterion = ‘entropy’, random_state = 0)<br>
classifier.fit(X_train, y_train)</p>
<h4 id="predicting-the-test-set-results-1">Predicting the Test set results</h4>
<p>y_pred = classifier.predict(X_test)</p>
<h4 id="making-the-confusion-matrix-2">Making the Confusion Matrix</h4>
<p>cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<h4 id="compute-metrics-6">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))<br>
cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<p>Acuracia do modelo é : 0.8048528652555498<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.84      0.92      0.88      1459
      1       0.65      0.46      0.54       478
</code></pre>
<p>avg / total       0.79      0.80      0.79      1937</p>
<p>Log Loss Result: 6.290849903880242<br>
Acuracia do modelo é : 0.804852865255549</p>
<p>Até então o pior modelo gerado, log loss alto, e acurácia baixa em relação aos outros modelos.<br>
A utilização do algorítimo Random Forest não foi satisfatório. A princípio os testes realizados com alteração de parâmetros não mostraram uma variância significativa. Desta forma descarto a priori o mesmo para este problema.</p>
<h1 id="novo-ajuste-no-dataset">Novo Ajuste no DataSet</h1>
<p>Diferente do teste anterior, o qual utilizei todas as colunas numéricas neste, irei utilizar apenas features identificadas no heatmap do processo de entendimento do data set como as com correlação mais forte. Este processo é uma tentativa para reduzir o log loss e melhorar a acurácia dos mesmos</p>
<h4 id="selecionando-features-com-base-no-heat-map">Selecionando features com base no heat map</h4>
<p>heat_corr(df)<br>
<img src="https://drive.google.com/file/d/1r3si58dLYNNX7ST8Ra8tox9QDD2soLxS/view?usp=sharing" alt="Heat Map Correlation"></p>
<p>#Selecionando features<br>
X = df[[‘age’,‘education-num’,‘hours-per-week’,‘capital-gain’]]<br>
y = df[‘col_pred’]</p>
<h3 id="svm-2">SVM</h3>
<h4 id="setup-the-pipeline-steps-steps-2">Setup the pipeline steps: steps</h4>
<p>steps = [(‘SVM’, SVC())]</p>
<h4 id="create-the-pipeline-pipeline-2">Create the pipeline: pipeline</h4>
<p>pipeline = Pipeline(steps)</p>
<h4 id="create-training-and-test-sets-2">Create training and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.45, random_state=30)</p>
<h4 id="feature-scaling-3">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="fit-the-pipeline-to-the-train-set-2">Fit the pipeline to the train set</h4>
<p>pipeline.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-4">Predict the labels of the test set</h4>
<p>y_pred = pipeline.predict(X_test)</p>
<h4 id="compute-metrics-7">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))<br>
cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<p>precision    recall  f1-score   support</p>
<pre><code>      0       0.82      0.96      0.89      1865
      1       0.77      0.39      0.52       625
</code></pre>
<p>avg / total       0.81      0.82      0.79      2490</p>
<p>Log Loss Result: 5.864679289632786<br>
Acuracia do modelo é : 0.8180722891566266</p>
<p>Modelo não obteve alterações significativas, nos testes realizados.<br>
Alterei percentual de split, random state em diversos valores, e não obtive resultado<br>
melhor do que encontrado no passo anterior com todas as colunas numéricas.</p>
<h3 id="logisticregression-2">LogisticRegression</h3>
<h4 id="create-the-hyperparameter-grid-1">Create the hyperparameter grid</h4>
<p>c_space = np.logspace(-15, 8, 15)<br>
param_grid = {‘C’: c_space, ‘penalty’: [‘l1’, ‘l2’]}</p>
<h4 id="instantiate-the-logistic-regression-classifier-logreg-2">Instantiate the logistic regression classifier: logreg</h4>
<p>logreg = LogisticRegression()</p>
<h4 id="create-train-and-test-sets-2">Create train and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)</p>
<h4 id="feature-scaling-4">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="instantiate-the-gridsearchcv-object-logreg_cv-1">Instantiate the GridSearchCV object: logreg_cv</h4>
<p>logreg_cv = GridSearchCV(logreg,param_grid,cv=5)</p>
<h4 id="fit-it-to-the-training-data-2">Fit it to the training data</h4>
<p>logreg_cv.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-5">Predict the labels of the test set</h4>
<p>y_pred = logreg_cv.predict(X_test)</p>
<h4 id="print-the-optimal-parameters-and-best-score-2">Print the optimal parameters and best score</h4>
<p>print(“Tuned Logistic Regression Parameter: {}”.format(logreg_cv.best_params_))<br>
print(“Tuned Logistic Regression Accuracy: {}”.format(logreg_cv.best_score_))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<h4 id="create-the-hyperparameter-grid-2">Create the hyperparameter grid</h4>
<p>c_space = np.logspace(-15, 8, 15)<br>
param_grid = {‘C’: c_space, ‘penalty’: [‘l1’, ‘l2’]}</p>
<h4 id="instantiate-the-logistic-regression-classifier-logreg-3">Instantiate the logistic regression classifier: logreg</h4>
<p>logreg = LogisticRegression()</p>
<h4 id="create-train-and-test-sets-3">Create train and test sets</h4>
<p>X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)</p>
<h4 id="feature-scaling-5">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="instantiate-the-gridsearchcv-object-logreg_cv-2">Instantiate the GridSearchCV object: logreg_cv</h4>
<p>logreg_cv = GridSearchCV(logreg,param_grid,cv=5)</p>
<h4 id="fit-it-to-the-training-data-3">Fit it to the training data</h4>
<p>logreg_cv.fit(X_train,y_train)</p>
<h4 id="predict-the-labels-of-the-test-set-6">Predict the labels of the test set</h4>
<p>y_pred = logreg_cv.predict(X_test)</p>
<h4 id="print-the-optimal-parameters-and-best-score-3">Print the optimal parameters and best score</h4>
<p>print(“Tuned Logistic Regression Parameter: {}”.format(logreg_cv.best_params_))<br>
print(“Tuned Logistic Regression Accuracy: {}”.format(logreg_cv.best_score_))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<p>Este modelo teve um resultado de log loss melhor do que o anterior, porém sua acurácia não foi tão boa quanto a com todas as features numéricas.</p>
<h3 id="random-forest-classification-2">Random Forest Classification</h3>
<p>Apesar dos últimos testes com RandomForest não terem sidos satisfatórios efetuarei a última tentativa com o mesmo. Considerando que o data set é diferente e o mesmo poderá ter um melhor desempenho devida a correlação entre as variáveis.</p>
<h4 id="splitting-the-dataset-into-the-training-set-and-test-set-3">Splitting the dataset into the Training set and Test set</h4>
<p>from sklearn.cross_validation import train_test_split<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)</p>
<h4 id="feature-scaling-6">Feature Scaling</h4>
<p>from sklearn.preprocessing import StandardScaler<br>
sc = StandardScaler()<br>
X_train = sc.fit_transform(X_train)<br>
X_test = sc.transform(X_test)</p>
<h4 id="fitting-random-forest-classification-to-the-training-set-2">Fitting Random Forest Classification to the Training set</h4>
<p>from sklearn.ensemble import RandomForestClassifier<br>
classifier = RandomForestClassifier(n_estimators = 10, criterion = ‘entropy’, random_state = 0)<br>
classifier.fit(X_train, y_train)</p>
<h4 id="predicting-the-test-set-results-2">Predicting the Test set results</h4>
<p>y_pred = classifier.predict(X_test)</p>
<h4 id="making-the-confusion-matrix-3">Making the Confusion Matrix</h4>
<p>cm = confusion_matrix(y_test, y_pred)<br>
ac_desemp(cm)</p>
<h4 id="compute-metrics-8">Compute metrics</h4>
<p>print(classification_report(y_test, y_pred))<br>
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))</p>
<p>Acuracia do modelo é : 0.7955601445534332<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.84      0.90      0.87      1459
      1       0.61      0.46      0.53       478
</code></pre>
<p>avg / total       0.78      0.80      0.79      1937</p>
<p>Log Loss Result: 6.590420082188497</p>
<p>Mais um vez o modelo apresentou resultados ruins, levando a descartá-lo totalmente<br>
para a solução deste problema analisado.</p>
<h2 id="conclusão-do-problema">Conclusão do Problema</h2>
<p>Com base nos testes possíveis de serem realizados no período definido para tal,<br>
considerei o modelo SVM utilizando todas as features numéricas do dataset como o<br>
com resultado mais aceitável.</p>
<p>O resultado a baixo reflete a executção do mesmo, obtendo uma acuracia de 0.829%<br>
com um log loss alto, porém o mais baixo encontrado nos testes realizados.</p>
<p>Existe a necessidade de se observar com mais tempo para poder efetuar mais testes e usar outras técnicas a fim de confirmar qual melhor modelo, provavelmente a aplicação da predição dos modelos com probability = True, torne-os com uma perda menor, reduzindo assim o log loss dos mesmos.</p>
<p>precision    recall  f1-score   supporthf</p>
<pre><code>      0       0.83      0.97      0.89      1033
      1       0.84      0.41      0.55       350
</code></pre>
<p>avg / total       0.83      0.83      0.81      1383</p>
<p>Log Loss Result: 5.50091361965321<br>
Acuracia do modelo é : 0.8293564714389009</p>

