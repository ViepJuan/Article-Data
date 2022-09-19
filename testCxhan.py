# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data viz. and EDA
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff

# # For scaling data
from mlxtend.preprocessing import minmax_scaling

# Tensorflow
#import tensorflow as tf

### Modelo de Tensorflow
# def build_model():
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(8, activation='relu', input_shape=[len(scaled_data.keys())]),
#     tf.keras.layers.Dense(4, activation='relu'),
#     tf.keras.layers.Dense(1,activation='sigmoid')
#   ])
#
#     optimizer = tf.keras.optimizers.RMSprop(0.01)
#
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
###
def correlation_plot():
    #correlation
    correlation = data.corr()
    #tick labels
    matrix_cols = correlation.columns.tolist()
    #convert to array
    corr_array  = np.array(correlation)
    trace = go.Heatmap(z = corr_array,
                       x = matrix_cols,
                       y = matrix_cols,
                       colorscale='blackbody',
                       colorbar   = dict()
                      )
    layout = go.Layout(dict(title = 'Matriz de Correlación',
                            #autosize = False,
                            #height  = 1400,
                            #width   = 1600,
                            margin  = dict(r = 0 ,l = 100,
                                           t = 0,b = 100,),
                            yaxis   = dict(tickfont = dict(size = 20)),
                            xaxis   = dict(tickfont = dict(size = 20)),
                           )
                      )
    fig = go.Figure(data = [trace],layout = layout)
    py.plot(fig)


## to find the median for filling null values
def find_median(var):
    temp = data[data[var].notnull()]
    temp = data[[var]].median().reset_index()
    return temp

def density_plot(var,size_bin):
    tmp1 = D[var]
    tmp2 = H[var]

    hist_data = [tmp1,tmp2]
    labels = ['Diabeties','Healthy']
    color = ['skyblue','indigo']
    fig = ff.create_distplot(hist_data,labels,colors = color,show_hist=True,bin_size=size_bin,curve_type='kde')

    fig['layout'].update(title = var)

    py.plot(fig, filename = 'Density plot')

## here I am using graph_obs as I am not able to costimize px.

def target_count():
    trace = go.Bar( x = data['Outcome'].value_counts().values.tolist(),
                    y = ['healthy','diabetic' ],
                    orientation = 'h',
                    text=data['Outcome'].value_counts().values.tolist(),
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.5,marker=dict(
                    color=['lightskyblue', ' indigo'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Count of affectes females')

    fig = dict(data = [trace], layout=layout)
    py.plot(fig)

# --------------- donut chart to show there percentage -------------------- #

def target_per():
    trace = go.Pie(labels=['healthy','diabetic' ],values=data['Outcome'].value_counts(),
                   textfont=dict(size=15),
                   opacity = 0.5, marker=dict(
                   colors=['lightskyblue','indigo'],line=dict(color='#000000', width=1.5)),
                   hole=0.6
                  )
    layout = dict(title='Donut chart to see the %age of affected.')
    fig = dict(data=[trace],layout=layout)
    py.plot(fig)


data = pd.read_csv("CXhan/in.csv")
# checking missing values if any
correlation_plot()
input()

# # lets see how many are affected by diabeties
D = data[data['Outcome'] == 1]
H = data[data['Outcome'] == 0]
# target_count()
# target_per()

data[['Edad','Peso','IMC','PA Sistolica','PA Diastolica',"Colesterol Total", "Colesterol LDL","Colesterol HDL","Trigliceridos" ]] = data[['Edad','Peso','IMC','PA Sistolica','PA Diastolica',"Colesterol Total", "Colesterol LDL","Colesterol HDL","Trigliceridos"]].replace("0",np.NaN)
data[['Edad','Peso','IMC','PA Sistolica','PA Diastolica',"Colesterol Total", "Colesterol LDL","Colesterol HDL","Trigliceridos" ]] = data[['Edad','Peso','IMC','PA Sistolica','PA Diastolica',"Colesterol Total", "Colesterol LDL","Colesterol HDL","Trigliceridos"]].replace(" ",np.NaN)
print(data.isnull().sum())

## Limpieza de Missing Values Para PIDD
missing=['Colesterol Total','Colesterol LDL','Colesterol HDL','Trigliceridos'] #Datos Faltantes
for miss in missing:
    promedios=find_median(miss)
    padece=round(promedios[0][0],2)
    print (padece)
    data.loc[(data[miss].isnull()), miss] = padece
    input()
## Termina Limpieza de Missing Values
print(data.info(),data.head())
correlation_plot()
#print(data.isnull().sum())
#density_plot('Insulin',0)
print(data.isnull().sum())

correlation_plot()
input("Presiona enter para escribir el archivo")
# ejecutar=input("Realizar Modelo? (Y/N)").lower()
# if (ejecutar=="y"):
#     scaled_data = minmax_scaling(data,columns=['Edad','Peso','IMC','PA Sistolica','PA Diastolica',"Gluc","HbA1c","Colesterol Total", "Colesterol LDL","Colesterol HDL","Trigliceridos","Creatinina" ])
#     model = build_model()
#     model.summary()
#     EPOCHS = 1000
#     history = model.fit(scaled_data, data['Outcome'],epochs=EPOCHS, validation_split=0.2, verbose=2)
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     acc = (hist['accuracy'].tail().sum())*100/5
#     val_acc = (hist['val_accuracy'].tail().sum())*100/5
#     print("Training Accuracy = {}% and Validation Accuracy= {}%".format(acc,val_acc))
data.round(2)
data.to_csv('out.csv')
