import pandas as pd
import numpy as np
import math

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score


import matplotlib.pyplot as plt
import randomcolor
from randomcolor import RandomColor

rand_color = randomcolor.RandomColor()


## Error Absoluto de las predicciones
def Error_point(y_pred, y):
    error = np.absolute(y_pred - y)
    return error


## Graficar en forma de matriz
def show_matrix_graph(num_columns, num_graph, type_graph, 
                      list_data_plot, 
                      data_predict, 
                      delta_dist, 
                      max_dist, 
                      save, save_name,
                      size_x,
                      size_y):
    
    plt.figure(figsize=(size_x,size_y))
    
    num_rows = math.ceil(len(list_data_plot)/num_graph) + 1
    
    cont = 0
    index = 1
    
    g_cant = []
    
    for i in range(1,num_rows+1,1):
        num_graph_sub = list_data_plot[cont:cont + num_graph]
        
        for g_type in num_graph_sub:
            color = rand_color.generate()[0]
            data_g = data_predict[data_predict[type_graph] == g_type]
            range_plot_model = data_range_(delta_dist, max_dist, data_g)
            
            cant_g = len(data_g)
            g_cant.append([g_type, cant_g])
            
            plt.subplot(num_rows,num_columns,index)
            
            plt.plot(range_plot_model['Range'].values,
                     range_plot_model['mean_RF'].values,
                     linewidth=0.5,
                     marker='o',
                     color=color,
                     markersize=5,
                     label=g_type)
            
            plt.fill_between(range_plot_model['Range'].values, 
                             range_plot_model['mean_RF'].values + range_plot_model['std_RF'].values, 
                             range_plot_model['mean_RF'].values - range_plot_model['std_RF'].values,
                             alpha=0.15, 
                             color=color)
            
            plt.title('Random Forest')
            plt.grid(linestyle='--')
            plt.legend(loc='upper left')
            plt.xlim(0,300)
            plt.ylim(0,300)
            plt.xlabel('Rangos (Km)')
            plt.ylabel('Error de predicción (s)')
            plt.yticks(range(0,300,25))
            
            plt.subplot(num_rows,num_columns,index+1)
            plt.plot(range_plot_model['Range'].values,
                     range_plot_model['mean_Pi'].values, 
                     linewidth=0.5, 
                     marker='o',
                     color=color,
                     markersize=5,
                     label=g_type)
            
            plt.fill_between(range_plot_model['Range'].values, 
                             range_plot_model['mean_Pi'].values + range_plot_model['std_Pi'].values, 
                             range_plot_model['mean_Pi'].values - range_plot_model['std_Pi'].values,
                             alpha=0.15, 
                             color=color)
            
            plt.title('Pildo')
            plt.grid(linestyle='--')
            plt.legend(loc='upper left')
            plt.xlim(0,300)
            plt.ylim(0,300)
            plt.xlabel('Rangos (Km)')
            plt.ylabel('Error de predicción (s)')
            plt.yticks(range(0,300,25));

            
                       
            cont += 1
        index += 2
    if save == 1:
        plt.savefig(save_name, dpi=150);
    
    return g_cant

## Obtenición de Aerolineas
def get_carries(callsign_gp):
    carrier_list = []
    carrier = pd.Series()
    for i in range(0, len(callsign_gp)):
        carrier_list.append(callsign_gp.values[i][:][0:3])
    carrier = pd.Series(np.array(carrier_list), name='Carrier')
    return carrier

## Obtención de los datos segun el rango deseado
def data_range_(delta_dist, max_dist, data_predict):
    data_range = pd.DataFrame()
    group_m = []
    
    # select groups
    for dist in range(0, max_dist, delta_dist):
        group_m = []       
        group = data_predict[ (data_predict['Dist'] > dist) & (data_predict['Dist'] < dist + delta_dist) ]
        
        count = len(group)
        
        #range_name =  str(dist/1000) +'-'+ str((dist + delta_dist)/1000)
        range_name =  (dist + delta_dist)/1000
        group_mean_RF = np.mean(group['Error_RF'].values)
        group_std_RF = np.std(group['Error_RF'].values)
        group_mean_Pi = np.mean(group['Error_Pi'].values)
        group_std_Pi = np.std(group['Error_Pi'].values)
        
        group_m.append([range_name, group_mean_RF, group_std_RF, group_mean_Pi, group_std_Pi, count])
        data_aux = pd.DataFrame(np.array(group_m), columns=list(['Range', 'mean_RF', 'std_RF', 'mean_Pi', 'std_Pi', 'Count']))
        
        data_range = pd.concat([data_range, data_aux], axis=0)
    
    return data_range

## Lectura de cantidad de vuelos
def read_flights(data, n_flights):
    ids_flights = np.unique(data['ID'].values)
    rand_flights_id = np.random.choice(ids_flights, size=n_flights, replace=False)
    data_f = pd.DataFrame()
    for ids in rand_flights_id:
        data_aux = data[data['ID'] == ids]
        data_f = pd.concat([data_f, data_aux])
    return data_f




## Definición del Estimador de Pildo
class PildoRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, demo_param='pildo'):
        self.demo_param = demo_param
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)
        predict_time = np.true_divide(X[:, 1], X[:, 0])
        return predict_time
    
    def score(self, X, y=None):
        return np.mean(np.absolute(self.predict(X) - y))
    
    




