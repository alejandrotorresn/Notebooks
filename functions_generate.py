import pandas as pd
import numpy as np
import math

## Error Absoluto de las predicciones
def Error_point(y_pred, y):
    error = np.absolute(y_pred - y)
    return error

## Obtenición de Aerolineas
def get_carries(callsign_gp):
    carrier_list = []
    carrier = pd.Series()
    for i in range(0, len(callsign_gp)):
        carrier_list.append(callsign_gp.values[i][:][0:3])
    carrier = pd.Series(np.array(carrier_list), name='Carrier')
    return carrier




## Lectura de cantidad de vuelos
def read_flights(data, n_flights):
    ids_flights = np.unique(data['ID'].values)
    rand_flights_id = np.random.choice(ids_flights, size=n_flights, replace=False)
    data_f = pd.DataFrame()
    for ids in rand_flights_id:
        data_aux = data[data['ID'] == ids]
        data_f = pd.concat([data_f, data_aux])
    return data_f