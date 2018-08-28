# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:25:35 2018

@author: filipe.luz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression


'''
Importando dataset do projeto
'''
#Importando do dataset do projeto
df = pd.read_csv('census.csv')

'''
Verificando registros e entendendo sua distribuição
'''
#Analisando dataset
df.info()
df.head()
df.tail()
df.shape
df.columns
df.describe()

#Verificação de NaN values
print('Quantidade de registros nulos em cada coluna do dataset:')
print(df.isnull().sum())


#Limpeza de NaN, existe uma linha de registro sem informações importantes bem como informação categorica
#O que gera uma dificuldade de aplicar uma categoriza e enviesar o modelo.
df = df.dropna()

#Verificação de NaN values apos excluir linhas com NaN
print('Quantidade de registros nulos em cada coluna do dataset, após limpeza:')
print(df.isnull().sum())
df.info()


def heat_corr (data):
    '''Analisando a correlação entre as features
       do dataset
    '''    
    sns.heatmap(data.corr(),square=True,cmap='RdYlGn')
    plt.title('Correlação entre as features')
    plt.show()


#Chamando função para analisar a correlação entre as features originais do dataset
heat_corr(df)    


'''
Criando Coluna para dizer se salário está a cima de 50k ou Não
Este processo torna-se necessário uma vez que o problema é classificar
de forma acertiva pessoas com salário a cima de 50. Algotirimos de classificação
trabalham melhor com features numéricas. 
'''
df['col_pred'] = np.where(df.income == '>50K', 1,0)


#Exibindo percentual de pessoas no dataset que estão 
#com salário a cima e a baixo de 50k existente no dataset
recs = df['age'].count()
qt_maior = (df.col_pred == 1).sum()
perc = round(float((qt_maior * 100) / recs),2)
print('Percentual de pessoas com salário acima de 50K: ' + str(perc) + ' %')
print('Percentual de pessoas com salário igual ou a baixo de 50K: ' + str(100 - perc) + ' %')

'''
Analisando correlação das features após a criação da coluna col_pred  
que terá a função de Y no nosso modelo de Classificação
'''
print('É possível analisar que algumas colunas \n possuem maior correlação com col_pred\n coluna originada da INCOME, após ajuste realizado. ') 
heat_corr(df)    


def distribuicao (data):
    '''
    Esta função exibirá a quantidade de registros únicos para cada coluna
    existente no dataset
    
    dataframe -> Histogram
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

#Chamando função para analisar a distribuição dos dados no data set
distribuicao(df)


def compute_log_loss(predicted, actual, eps=1e-14):
    '''
    Computa a medida de avaliação sobre perda (log loss)
    Utilizada para avaliação do modelo
    '''
    
    predicted = np.clip( predicted , eps , 1- eps)  
    loss = -1 * np.mean(  actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    return loss


def ac_desemp (cm):
    ac = 0
    ac = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[1,0] + cm[0,1])
    
    print("Acuracia do modelo é : " + str(ac))

'''================================== SPLIT DO DATA SET =========================================='''

'''Excluindo feature income, essa feature por ser descritiva 
   diminui o desempenho do algoritimo de classificação, utilizaremos
   a coluna col_pred criada anteriormente. 
'''

#Excluindo coluna income e col_pred, para gerar X 
X = df.drop(['col_pred','income'],axis=1)

#Separando coluna col_pred, a qual será nossa Y
y = df['col_pred']

'''
No passo a baixo, estou separando colunas numéricas das não numéricas
com o intuito de posteriormente poder avaliar os modelos apenas com colunas numéricas
e para poder realizar processos de tratamento nas features de texto.
'''
#Buscando colunas descritivas do data set 
LABELS = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] in ['object']]
NUMB = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] not in ['object']]

'''
Foi escolhida a tecnica de Dummy columns, com exclusão de uma coluna para cada 
grupo de campo descritivo, seguindo melhores práticas.
O método dumy gera colunas para cada campo descritivo do data set,
por Ex.: Estado: DF,SP,RJ. Será criada coluna Estado_DF,Estado_RJ. Quando o registro for
DF a coluna Estado_DF terá valor 1, Estado_RJ terá valor 0. Quando Estado_df e Estado_rj 
forem respectivamente 0, significa que o Estado é SP.
Este processo é feito para melhorar a classificação dos algoritimos que usam medida euclidiana.
Foi escolhido no lugar do Hot Encoder pois facilita o processo de ajuste do dataset
apesar de aumentar significativamente a quantidade de colunas.
'''

#Aplicando tecnica dummy nas colunas descritivas
X_dum = pd.get_dummies(X[LABELS], drop_first=True)

print('Colunas do dataset após tecnica dummy')
print(X_dum.columns)
print('Detakhamento do dataset após tecnica dummy')
print(X_dum.info())



'''========================================MODELOS DE CLASSIFICAÇÃO========================================================='''
'''
Após os ajustes no dataset iremos aplicar os modelos de ML
pra classificar os registros do dataset, ensinando assim a 
maquina a classificar de forma mais acertiva possível novos registros, posteriormente
ao aprendizado.

'''

'''SVM MODEL COM TODO DATASET, USANDO AS DUMMY COLUMNS'''

'''=======Suport Vector Machine ====='''

# Setup the pipeline steps: steps
steps = [('SVM', SVC())]
# Create the pipeline: pipeline
pipeline = Pipeline(steps)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_dum,y, test_size=0.25, random_state=0)
# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)
# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)


'''============LogisticRegression=============='''
'''
Para tentar melhorar o desempenho do SVM utilizei o método de tuning
Grid Search, o qual verifica o desempenho do algoritimo com diferentes parametos selecionados
retornando os parâmetros que obtiveram o melhor desempenho.
Utilizarei as principais opções de tuning do algoritimo, levando em consideração
o processo de regularização dos dados, como opção l1 = Lasso e l2 = Ridge
Acredito que a utilização de regularização nesse modelo faz dele candidato a modelo ideal para o problema.

'''

# Criação da grade de hiperparametros
c_space = np.logspace(-5, 8, 45) 
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_dum,y,test_size=0.25,random_state = 0)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = logreg_cv.predict(X_test)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))


'''===============================Random Forest Classification ======================'''
'''
Utilizarei o Random Forest por ser um algritimo com alto potêncial de classificação
apesar de ter o risco de muitos falsos positivos é um algorítimo
muito utilizado pela comunidade, obtendo resultados significativos em competições de Data Science
Por ser uma junção de Decision Trees pode ser que consigamos um melhor desempenho na classificação
do dataset utilizado.
'''


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dum, y, test_size = 0.25,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))

'''============================K-NN====================================='''
'''
O K-NN é recomendado para modelos de classificação, utiliza similaridade dos dados
para promover a classificação desejada.
'''


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dum, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting KNN Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors = 5,
                                  metric = 'minkowski',
                                  p = 2 )
classifier.fit(X_train, y_train)

#Predict the Test set Results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)
# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))


'''============================= Teste dos Modelos com Todas colunas numericas ========================'''
'''
Após executar os modelos com o data set completo e identificar que os mesmos estavam
estagnando em relação a acurácia e log loss alto, resolvi testar os algorítimos utilizando
apenas as features numéricas com maior correlação. Usei como Base o heatmap gerado nos
passos iniciais para selecionar as features que seriam utilizadas nos testes seguintes.
'''
#Separando dataset com apenas colunas numéricas
X_num = X[NUMB]

#Verificando a escala das colunas numéricas
print("Média das medidas sem escala: {}".format(np.mean(X_num))) 
print("Desvio Padrão das medidas sem escala: {}".format(np.std(X_num)))

'''
Aplicando método scalling para aproximar os valores das colunas numéricas
a fim de tentar melhorar o desempenho das classificações uma vez que medidas muito
espaçadas inferem no resultado do algorítimo.
'''
from sklearn.preprocessing import scale
#Scalling
X_scaled = scale(X_num)


# Print the mean and standard deviation of the scaled features
print("Média das medidas com escala: {}".format(np.mean(X_scaled))) 
print("Desvio Padrão das medidas com escala: {}".format(np.std(X_scaled)))


'''SVM com colunas numéricas e ajustadas via scalling, objetivo é baixar o log loss
  cMantendo a acurácia do modelo de classificação
'''

'''
Preparando para criar modelo em SVM
'''

# Setup the pipeline steps: steps
steps = [('SVM', SVC())]
# Create the pipeline: pipeline
pipeline = Pipeline(steps)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.25, random_state=30)
# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)
# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)



'''
O resultado apresentado pelo SVM com escala foi satisfatório em relação
a melhora da acurácia do modelo, porém a tentativa de baixar o log loss não se confirmou
neste modelo.
'''

'''============LogisticRegression=============='''


# Create the hyperparameter grid
c_space = np.logspace(-15, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}


# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.25,random_state = 30)


# Instantiate the GridSearchCV object: logreg_cv
#Na tentativa de optimizar o processo de tuning, aumentei a quantidade de verificações
#Passanco cv de 5 para 10 tentativas
logreg_cv = GridSearchCV(logreg,param_grid,cv=10)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = logreg_cv.predict(X_test)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_test,y_pred)))
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)


'''===============================Random Forest Classification ======================'''
'''
Aplicando RandomForest com colunas scalonadas e dataset apenas numérico.

'''


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.35, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)

'''
A utilização do algoritimo Random Forest não foi satifatório. A pricípio os testes realizados com
alteração de parametros não mostraram uma variancia significativa. Desta forma descarto apriori 
o mesmo para este problema. 
'''


'''============================K-NN====================================='''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.35, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting KNN Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors = 5,
                                  metric = 'minkowski',
                                  p = 2 )
classifier.fit(X_train, y_train)

#Predict the Test set Results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)
# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))

'''
Os testes com este algoritimo também não foram satisfatórios, mesmo aplicando regularização ou não, permanecendo 
o padrão para metrica minkowski, o algorítimo não variou significativamente seu resultado de classificação.
Levando também em conta a validação com log loss que em todos os casos de modelos utilizados
não foi satisfatório, trabalhando sempre em uma margem de 5, quando o menor valor possível mostraria
um melhor modelo e mais confiável.

'''



'''====================================NOVO AJUSTE DO DATA SET =================================='''
''' Diferente do teste anterior, o qual utilizei todas as colunas numéricas neste, irei utilizar apenas features 
indentificadas no heatmap do processo de entendimento do dat set como as com correlação mais forte.
Este processo é uma tentativa para reduzir o log loss e melhorar a acuracia dos mesmos'''

'''================================== 3 º SPLIT DO DATA SET =========================================='''
'''
Selecionando features com base no heat map
'''
heat_corr(df)    

#Selecionando features
X = df[['age','education-num','hours-per-week','capital-gain']]
y = df['col_pred']

'''==============================================SVM====================================='''
'''
Preparando para criar modelo em SVM
'''

# Setup the pipeline steps: steps
steps = [('SVM', SVC())]
# Create the pipeline: pipeline
pipeline = Pipeline(steps)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.45, random_state=30)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)
# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)

'''
Modelo não obteve alterações significativas, nos testes realizados.
Alterei percentual de split, random state em diversos valores, e não obtive resultado
melhor do que encontrado no passo anterior com todas as colunas numéricas.

'''


'''============================LogisticRegression===================================='''


# Create the hyperparameter grid
c_space = np.logspace(-15, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = logreg_cv.predict(X_test)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))

'''
Este modelo teve um resultado de log loss melhor do que o anterior, porém sua acuracia não foi
tão boa quanto a com todas as features numéricas.
'''


'''===============================Random Forest Classification ======================'''
'''
Apesar dos últimos testes com RandomForest não terem sidos satisfatórios efetuarei a ultima
tentativa com o mesmo. Considerando que o data set é diferente e o mesmo poderá ter um melhor desempenho
devida a correlação entre as variáveis.
'''


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)

# Compute metrics
print(classification_report(y_test, y_pred))
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))

'''
Mais um vez o modelo apresentou resultados ruins, levando a descartá-lo totalmente
para a solução deste problema analisado.
'''



'''============================K-NN====================================='''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 80)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting KNN Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors = 80,
                                  metric = 'minkowski',
                                  p =2 )
classifier.fit(X_train, y_train)

#Predict the Test set Results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac_desemp(cm)
print("Log Loss Result: " + str( compute_log_loss(y_pred,y_test)))


'''
O modelo KNN teve um desempenho aceitável, porém não foi o melhor dos modelos utilizados
o que se pode avaliar é que a diferença nas classificações com as features numéricas
foram significativas neste modelo. Foram tentadas algumas opções de ajustes do modelo.

'''

'''====================================Conclusão do Problema ===================================='''
'''
Com base nos testes possíveis de serem realizados no período definido para tal,
considerei o modelo SVM utilizando todas as features numéricas do dataset como o 
com resultado mais aceitável.

O resultado a baixo reflete a executção do mesmo, obtendo uma acuracia de 0.829%
com um log loss alto, porém o mais baixo encontrado nos testes realizados.

Existe a necessidade de se observar com mais tempo para poder efetuar mais testes e usar ouutras técnicas
a fim de confirmar qual melhor modelo.


 precision    recall  f1-score   support

          0       0.83      0.97      0.89      1033
          1       0.84      0.41      0.55       350

avg / total       0.83      0.83      0.81      1383

Log Loss Result: 5.50091361965321
Acuracia do modelo é : 0.8293564714389009
'''


