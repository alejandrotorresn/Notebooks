import pandas as pd
import numpy as np
import math
from scipy.spatial import distance




### A. Convertir las coordenadas GPS (°, ', '') a coordenadas GPS decimales

# Los parámetros de ingreso estan en formato GGMMSSD
# donde 
#      GG -> Grados
#      MM -> Minutos
#      SS -> Segundos
#      D -> Dirección
#
#  Ejemplo: 473651N
#
# Retorna Latitude y Longitud en GPS Decimales


def Tbar_convert(lat, lon):
    lat = lat.strip()
    lat_split = []
    while lat:
        lat_split.append(lat[:2])
        lat = lat[2:]
    N = 'N' in lat_split[3]
    d, m, s = float(lat_split[0]), float(lat_split[1]), float(lat_split[2])
    Latitude = (d + m / 60. + s / 3600.) * (1 if N else -1)
        
    lon = lon.strip()
    lon = lon[1:]
    lon_split = []
    while lon:
        lon_split.append(lon[:2])
        lon = lon[2:]
    W = 'W' in lon_split[3]
    d, m, s = float(lon_split[0]), float(lon_split[1]), float(lon_split[2])
    Longitude = (d + m / 60. + s / 3600.) * (-1 if W else 1)
            
    return Latitude, Longitude


### B. Convertir coordenadas decimales a coordenadas Web Mercator

# Los parámetros de ingreso estan en formato GPS decimal, en valor único o como una lista de cada parámetro
# Ejemplo:
#         lat: 47.1234
#         lon: 19.345
# ó
#         lat: [47.122, 47.343, 47.988]
#         lon: [19.455, 19.983, 19.738]
#
# Retorna Lat_y y Lon_x en metros

def geographic_to_web_mercator(lat, lon):
    
    num = np.radians(lon)
    Lon_x = 6378137.0 * num
    
    a = np.radians(lat)
    Lat_y = 3189068.5 * np.log((1.0 + np.sin(a)) / (1.0 - np.sin(a)))
    
    coord = (np.stack((Lat_y, Lon_x), axis=-1))
    
    cond_1 = np.less_equal(np.absolute(lon), 180)
    cond_2 = np.less(np.absolute(lat), 90)
    
    result_cond = np.logical_and(cond_1, cond_2)
    
    return result_cond, coord


### C. Convertir las coordendas del dataset de las Tbar a coordenadas Web Mercator
# Retorna un Dataframe con las coordendas en metros para los puntos de las Tbars

def points_tbars(data_tbars):
    
    coord_tbars = []
    tbar_names = data_tbars['name'].drop_duplicates().values
    
    for tbar_name in tbar_names:
        data_tb = data_tbars[data_tbars['name'] == tbar_name]
        ids_tbar = data_tb['id'].drop_duplicates().values
        for id_tbar in ids_tbar:
            lat_id = data_tb[data_tb['id'] == id_tbar]['wlat'].values[0]
            lon_id = data_tb[data_tb['id'] == id_tbar]['wlon'].values[0]
            # Convertir las coordenadas GPS (°,','') a coordenadas decimales
            lat_id, lon_id = Tbar_convert(lat_id, lon_id)
            # Convertir las coordenads GPS decimales a coordenadas Web Mercator          
            result_cond, coord = geographic_to_web_mercator(lat_id, lon_id)
            coord_tbars.append([tbar_name, id_tbar, lat_id, lon_id, coord[0], coord[1]])
    
    coord_df = pd.DataFrame(np.array(coord_tbars), columns=list(['name', 'id', 'lat', 'lon', 'wlat', 'wlon']))
    
    return coord_df


### D. Crea retorna las coordenadas de los puntos de las Tbars
def Tbar_graph(tbar):
    point_center = []
    point_left = []
    point_right = []
    point_rwy = []
    
    for tbar_point in tbar:
        lon_tbar, lat_tbar = geographic_to_web_mercator(tbar_point[2], tbar_point[1])
        if tbar_point[0] == 'center':
            point_center = lat_tbar, lon_tbar
        if tbar_point[0] == 'left':
            point_left = lat_tbar, lon_tbar
        if tbar_point[0] == 'right':
            point_right = lat_tbar, lon_tbar
        if tbar_point[0] == 'rwy':
            point_rwy = lat_tbar, lon_tbar
            
    return point_center, point_left, point_right, point_rwy



### E. Búsqueda del punto central
def central_point(points_tb):
    central_p = []
    points_rwy = points_tb[points_tb['id'] == 'rwy']
    if len(points_rwy) == 1:
        central_p = (points_rwy[['lat', 'lon', 'wlat', 'wlon']]).reset_index(drop=True)
    else:
        central_p_lat = np.mean((points_rwy['lat'].values.astype(float)))
        central_p_lon = np.mean((points_rwy['lon'].values.astype(float)))
        
        result_cond, coord = geographic_to_web_mercator(central_p_lat, central_p_lon)
        central_p.append([central_p_lat, central_p_lon, coord[0], coord[1]])

        central_p = pd.DataFrame(np.array(central_p), columns=list(['lat', 'lon', 'wlat', 'wlon']))
    return central_p


### F. Radio máximo
def radio_max(central_p, points_tb):

    tbar_names = points_tb['name'].drop_duplicates().values
    
    points_left = points_tb[points_tb['id'] == 'left']
    points_right = points_tb[points_tb['id'] == 'right']
    
    x1 = central_p['wlon'].values.astype(float)[0]
    y1 = central_p['wlat'].values.astype(float)[0]
    
    if len(tbar_names) == 1:
        
        x2 = points_left['wlon'].values.astype(float)[0]
        y2 = points_left['wlat'].values.astype(float)[0]
        
        radio = distance.euclidean([x1, y1], [x2, y2])
        
    else:
        
        x2 = points_left['wlon'].values.astype(float)
        y2 = points_left['wlat'].values.astype(float)
                
        radio = 0
        
        for i in range(len(x2)):
            radio_m = distance.euclidean([x1, y1], [x2[i], y2[i]])
            if radio_m > radio:
                radio = radio_m
                
    return radio


### G. Coordenadas GPS a coordenadas Web Mercator
def GPS_to_WebMercator(data):
    lon = data['LONGITUDE'].values
    lat = data['LATITUDE'].values
    
    result_cond, coord = geographic_to_web_mercator(lat, lon)
    data_aux = pd.DataFrame(np.array(coord), columns=list(['LATITUDE_meters', 'LONGITUDE_meters']))
    data = pd.concat([data, data_aux], axis=1).copy(deep=True)
    
    points_bef = len(data)
    
    result_cond = result_cond.tolist()
    data = data[result_cond].copy(deep=True)
    
    points_aft = len(data)
    data = data.reset_index(drop=True)
    
    return data    

### H. Filtrado de puntos fuera del radio r
def filter_points_radius(data, central_p, radius):
    x1 = central_p.wlon
    y1 = central_p.wlat
    
    x2 = data['LONGITUDE_meters'].values
    y2 = data['LATITUDE_meters'].values
    
    p1 = np.stack((x1, y1), axis=-1)
    p1 = p1.reshape(1,2)
    p2 = np.stack((x2, y2), axis=-1)
    
    dist = distance.cdist(p1, p2, 'euclidean')
    
    index = dist < radius
    
    data = data[index[0]]
    
    return data

### I. Búsqueda de los puntos finales de cada trayectoría
def search_Fpoint(data):
    
    id_flights = data['ID'].drop_duplicates().values
    flight_Fp = []
    
    for id_flight in id_flights:
        # Time Max
        time = data[data['ID'] == id_flight]['TIMESTAMP']
        time_max = int(time.max())
        
        # Lat and Lon Max
        data_flight = data[data['ID'] == id_flight]['TIMESTAMP'] == time_max
        data_flight = data[data['ID'] == id_flight][data_flight]
        lat = data_flight['LATITUDE_meters'].values[0]
        lon = data_flight['LONGITUDE_meters'].values[0]
        
        flight_Fp.append([id_flight, time_max, lat, lon])
            
    Fpoint_flights = pd.DataFrame(np.array(flight_Fp), columns=list(['ID', 'TIMESTAMP', 'LATITUDE_meters', 'LONGITUDE_meters']))
    Fpoint_flights['ID'] = Fpoint_flights['ID'].astype('int64')
    
    return Fpoint_flights


### J. Búsqueda de rutas que tengan el punto final dentro del radio establecido
def validate_Fp(radio, central_p, flights_Fp):
    
    flights = []
    
    a = central_p['wlon'].values.astype(float)[0]
    b = central_p['wlat'].values.astype(float)[0]

    id_flights = flights_Fp['ID'].drop_duplicates().values
    
    for id_flight in id_flights:
        x = flights_Fp[flights_Fp['ID'] == id_flight]['LONGITUDE_meters']
        y = flights_Fp[flights_Fp['ID'] == id_flight]['LATITUDE_meters']
        
        eq = np.square(x - a) + np.square(y - b) - np.square(radio)
                
        if (eq.values <= 0):
            flights.append(True)
        else:
            flights.append(False)
        
    return flights

### K. Búsqueda de la Tbar correspondiente a cada vuelo
# flights_Fp    ->  punto final de cada uno de los vuelos
# points_tb     ->   coordenadas de las Tbars

def select_Tbar(flights_Fp, points_tb):
    
    names_Tbar = points_tb['name'].drop_duplicates().values
    points_rwy = points_tb[points_tb['id'] == 'rwy']
    
    dist_Tbar_Fp = pd.DataFrame()
    
    if len(names_Tbar) == 1:
        return (names_Tbar)
    else:
        for name_Tbar in names_Tbar:
            x1 = flights_Fp['LONGITUDE_meters'].values
            y1 = flights_Fp['LATITUDE_meters'].values
            
            x2 = points_rwy[points_rwy['name'] == name_Tbar]['wlon'].values[0]
            y2 = points_rwy[points_rwy['name'] == name_Tbar]['wlat'].values[0]            
            
            p1 = np.stack((x1, y1), axis=-1)
            p2 = np.stack((x2, y2), axis=-1)
            p2 = p2.reshape(1,2)
            dist = distance.cdist(p1, p2, 'euclidean')
                        
            dist_aux = pd.DataFrame(np.array(dist), columns=list([name_Tbar]))            
            dist_Tbar_Fp = pd.concat([dist_Tbar_Fp, dist_aux], axis=1)
            
            
        list_Tbar_names = []
        id_flights = flights_Fp['ID'].drop_duplicates().values
        list_Tbar_names = dist_Tbar_Fp.isin(dist_Tbar_Fp.min(axis=1)).idxmax(axis=1)[:]
        list_Tbar_names = np.array(list_Tbar_names)
        list_Tbar_names = np.stack((id_flights, list_Tbar_names), axis=-1)
        list_pd = pd.DataFrame(list_Tbar_names, columns=list(['ID', 'Tbar']) )
        list_pd['ID'] = list_pd['ID'].astype('int64')
        
        return list_pd
    
    return 0    


### L. Ecuaciones de las rectas para las diferentes Tbars
def eq_line(points_tb, name_Tbar, x, y):
    Tbar = points_tb[points_tb['name'] == name_Tbar]
    x1 = Tbar[Tbar['id'] == 'left']['wlon'].astype(float).values[0]
    y1 = Tbar[Tbar['id'] == 'left']['wlat'].astype(float).values[0]
    x2 = Tbar[Tbar['id'] == 'right']['wlon'].astype(float).values[0]
    y2 = Tbar[Tbar['id'] == 'right']['wlat'].astype(float).values[0]
    
    m = (y2 - y1)/(x2 - x1)
    eq = m*x - m*x1 - y + y1
        
    return eq

## M. Calcular angulo entre dos vectores
def calc_angle(Vector_x, Vector_y):
    Vector_x = np.where(Vector_x > 0, Vector_x, 1)
    divide = np.true_divide(Vector_y,Vector_x)
    angle = (np.arctan(divide))
    return angle

## N.  Calcular la proyección ortogonal
def ortho_projection(W, V):
    #Vp = (V.dot(W)/V.dot(V))*V
    Vp = (V.dot(W)/V.dot(V))
    #Vpv = np.sqrt(np.square(Vp[0]) + np.square(Vp[1]))
    
    
    #W[0] = np.where(W[0] > 0, W[0], 1)
    #divide = np.true_divide(W[1],W[0])
    #angle = (np.arctan(divide))
    
    #return Vpv, angle
    return Vp


## O. Lectura de vuelos

def read_flights(data, n_flights):
    ids_flights = np.unique(data['ID'].values)
    rand_flights_id = np.random.choice(ids_flights, size=n_flights, replace=False)
    data_f = pd.DataFrame()
    for ids in rand_flights_id:
        data_aux = data[data['ID'] == ids]
        data_f = pd.concat([data_f, data_aux])
    return data_f



## P.Conversión de la velocidad de Knots a m/s
def convert_knot_ms(velocity):
    factor = 1.943844
    velocity = np.divide(velocity, factor)
    return velocity




def Error_point(y_pred, y):
    error = np.absolute(y_pred - y)
    return error



def data_range_(delta_dist, max_dist, data_predict):
    data_range = pd.DataFrame()
    group_m = []
    
    # select groups
    for dist in range(0, max_dist, delta_dist):
        group_m = []       
        group = data_predict[ (data_predict['Dist'] > dist) & (data_predict['Dist'] < dist + delta_dist) ]
        
        count = len(group)
        
        range_name =  (dist + delta_dist)/1000
        group_mean_RF = np.mean(group['Error'].values)
        group_std_RF = np.std(group['Error'].values)
        
        group_m.append([range_name, group_mean_RF, group_std_RF, count])
        data_aux = pd.DataFrame(np.array(group_m), columns=list(['Range', 'mean_RF', 'std_RF', 'Count']))
        
        data_range = pd.concat([data_range, data_aux], axis=0)
    
    return data_range


def transform_dataset(path_tbar, path_dataset, radio_filter, T):
    data_tbars = pd.read_csv(path_tbar)
    points_tb = points_tbars(data_tbars)
    central_p = central_point(points_tb)
    radio = radio_max(central_p, points_tb)
    data = pd.read_csv(path_dataset)
    data = GPS_to_WebMercator(data)
    data = filter_points_radius(data, central_p, radio_filter)
    data = data.reset_index(drop=True)

    flights_Fp = search_Fpoint(data)
    # que pasa si el vuelo no esta dentro del radio
    flights_valid = validate_Fp(radio, central_p, flights_Fp)
    flights_Fp = flights_Fp[flights_valid]

    ids = flights_Fp['ID'].values
    data = data[data.ID.isin(ids)].copy(deep=False)
    data = data.reset_index(drop=True)

    Tbar_direction = select_Tbar(flights_Fp, points_tb)


    id_flights = data['ID'].drop_duplicates().values
    Tbar = []

    for id_flight in id_flights:
        data_aux = len(data[data['ID'] == id_flight])
        name_Tbar = Tbar_direction[Tbar_direction['ID'] == id_flight]['Tbar']
        Tbar = Tbar + ([name_Tbar.values[0] for x in range(data_aux)])

    Tbar_df = pd.DataFrame(np.asarray(Tbar), columns=list(['Tbar']))
    Tbar_df = Tbar_df.reset_index(drop=True)
    data = pd.concat([data, Tbar_df], axis=1)


    flight_bool = []
    point_data = pd.DataFrame()

    names_Tbar = points_tb['name'].drop_duplicates().values

    for name_Tbar in names_Tbar:

        data_flightsTbar = data[data['Tbar'] == name_Tbar]
        id_flights = data_flightsTbar['ID'].drop_duplicates().values

        for id_flight in id_flights:

            x = data_flightsTbar[data_flightsTbar['ID'] == id_flight]['LONGITUDE_meters'].values
            y = data_flightsTbar[data_flightsTbar['ID'] == id_flight]['LATITUDE_meters'].values

            eq = eq_line(points_tb, name_Tbar, x, y)

            if len(eq) > 0 :
                eq = (np.diff(np.sign(eq)) != 0)*1   # pone todo a cero y un 1 donde se produce el cambio de indice
                ind = np.argmax(eq)  # Retorna el indice donde se encuentra el punto mas cercano a la línea
                                     # Este indice es con respecto al vector que se extrajo para cada vuelo
                data_flight = data[data['ID'] == id_flight][['ID', 'LATITUDE_meters', 'LONGITUDE_meters', 'TIMESTAMP', 'Tbar']]

                test = data_flight.iloc[ind]
                test = test.to_frame().T

                point_data = pd.concat([point_data, test], axis=0)

                # El indice tiene que ser mayor a cero            
                if (ind > 0):
                    flight_bool.append(True)
                else:
                    flight_bool.append(False)

    point_data = point_data.reset_index(drop=True)
    point_data['ID'] = point_data['ID'].astype('int64')
    point_data = point_data[flight_bool]
    point_data = point_data.reset_index(drop=True)        


    ids = point_data['ID'].values
    data = data[data.ID.isin(ids)].copy(deep=False)
    data = data.reset_index(drop=True)

    data = data.iloc[list( i for i in range(0, len(data), T))]
    data = data.reset_index(drop=True)

    ids = np.unique(data['ID'].values)

    cont_flights = 0
    data_new = pd.DataFrame()
    factor = 2

    for id_f in ids:
        data_id = data[data['ID'] == id_f]

        lon_mean = data_id.LONGITUDE_meters.mean()
        lon_std = data_id.LONGITUDE_meters.std()    

        for point_lon in data_id.LONGITUDE_meters:
            if point_lon > (lon_mean + factor*lon_std):
                data_id = data_id.drop(data_id[data_id.LONGITUDE_meters == point_lon].index )
            if point_lon < (lon_mean - factor*lon_std):
                data_id = data_id.drop(data_id[data_id.LONGITUDE_meters == point_lon].index )

        data_new = pd.concat([data_new, data_id])

    data_new = data_new.reset_index(drop=True)        

    data = data_new.copy()
    del data_new

    mean_flights_points = int(np.trunc(np.unique(((data.groupby('ID').count()).mean()).values)))

    for id_f in ids:
        long = len(data[data['ID'] == id_f])
        if long < mean_flights_points:
            data = data.drop(data[data['ID'] == id_f].index)
            cont_flights += 1        


    data = data.reset_index(drop=True)

    # para cada ID sacar la distacia euclidea entre el punto y su punto de intercepcion

    id_flights =  data['ID'].drop_duplicates().values

    distTbar = pd.DataFrame()
    dist = 0
    for id_flight in id_flights:

        x1 = data[data['ID'] == id_flight]['LONGITUDE_meters'].values
        y1 = data[data['ID'] == id_flight]['LATITUDE_meters'].values


        x2 = point_data[point_data['ID'] == id_flight]['LONGITUDE_meters'].values[0]
        y2 = point_data[point_data['ID'] == id_flight]['LATITUDE_meters'].values[0]


        p1 = np.stack((x1, y1), axis=-1)
        p2 = np.stack((x2, y2), axis=-1)
        p2 = p2.reshape(1,2)
        dist = distance.cdist(p1, p2, 'euclidean')

        distAux = pd.DataFrame(np.asarray(dist), columns=list(['dist_Tbar']))            
        distTbar = pd.concat([distTbar, distAux], axis=0)

    distTbar = distTbar.reset_index(drop=True)


    data = pd.concat([data, distTbar], axis=1)
    data = data.reset_index(drop=True)


    Vx = data['VELOCITY_EW'].values
    Vy = data['VELOCITY_NS'].values
    V = (np.sqrt(np.square(Vx) + np.square(Vy)))
    angleV = calc_angle(Vx, Vy)
    Vel = np.stack((V, angleV), axis=-1)    
    Vel = pd.DataFrame(Vel, columns=list(['Velocity', 'Angle_V']))
    Vel = Vel.reset_index(drop=True)


    data = pd.concat([data, Vel], axis=1)
    data = data.reset_index(drop=True)

    Ax = data['ACCELERATION_EW'].values
    Ay = data['ACCELERATION_NS'].values
    A = (np.sqrt(np.square(Ax) + np.square(Ay)))
    angleA = calc_angle(Ax, Ay)
    Acc = np.stack((A, angleA), axis=-1)    
    Acc = pd.DataFrame(Acc, columns=list(['Acceleration', 'Angle_A']))
    Acc = Acc.reset_index(drop=True)

    data = pd.concat([data, Acc], axis=1)
    data = data.reset_index(drop=True)

    V = data[['VELOCITY_EW', 'VELOCITY_NS']].values
    W = data[['WIND_EW', 'WIND_NS']].values
    Vpv = []
    for val in range(len(V)):
        Vpv.append(ortho_projection(W[val],V[val]))


    #Vp = pd.DataFrame(Vpv, columns=list(['Velocity_p', 'Angle_p']))
    Vp = pd.DataFrame(Vpv, columns=list(['Velocity_p']))
    data = pd.concat([data, Vp], axis=1)
    data = data.reset_index(drop=True)


    # para cada ID sacar el delta entre el tiempo de ese punto y su punto de intercepcion
    id_flights =  data['ID'].drop_duplicates().values

    point_target = []
    point_limit = []

    for id_flight in id_flights:

        time = data[data['ID'] == id_flight]['TIMESTAMP'].values
        time_base = point_data[point_data['ID'] == id_flight]['TIMESTAMP'].values

        points_valid = np.less(time, time_base)
        subtract = (np.subtract(time,time_base)/1000000) * -1

        point_limit = point_limit + points_valid.tolist()
        point_target = point_target + subtract.tolist()

    target = pd.DataFrame(np.asarray(point_target), columns=list(['Target_sec']))        


    data = pd.concat([data, target], axis=1)
    data = data.reset_index(drop=True)

    data = data[point_limit]
    data = data.reset_index(drop=True)

    data['Velocity'] = convert_knot_ms(data['Velocity'].values)

    dataset = data[['ID', 
                    'CALLSIGN', 
                    'AC_REAL_MODEL',
                    'Tbar',
                    'LATITUDE_meters', 
                    'LONGITUDE_meters', 
                    'Velocity', 
                    'Angle_V', 
                    'Acceleration', 
                    'Angle_A', 
                    'dist_Tbar',
                    'Velocity_p',
                    'Target_sec']]

    return dataset


def transform_point(name_tbar, point_tbar, path_tbar, path_point):
    
    data_tbars = pd.read_csv(path_tbar)
    data_tbars = data_tbars[data_tbars['name'] == name_tbar]
    
    points_tb = points_tbars(data_tbars)    
    central_p = central_point(points_tb)
    radio = radio_max(central_p, points_tb)
    
    data = pd.read_csv(path_point)
    data = GPS_to_WebMercator(data)
    
    Tbar = [name_tbar]
    Tbar_df = pd.DataFrame(np.asarray(Tbar), columns=list(['Tbar']))
    Tbar_df = Tbar_df.reset_index(drop=True)
    data = pd.concat([data, Tbar_df], axis=1)
    
    distTbar = pd.DataFrame()
    dist = 0
    
    x1 = data['LONGITUDE_meters'].values
    y1 = data['LATITUDE_meters'].values
    
    x2 = points_tb[points_tb['id'] == point_tbar]['wlon'].values[0]
    y2 = points_tb[points_tb['id'] == point_tbar]['wlat'].values[0]


    p1 = np.stack((x1, y1), axis=-1)
    p2 = np.stack((x2, y2), axis=-1)
    p2 = p2.reshape(1,2)
    dist = distance.cdist(p1, p2, 'euclidean')

    distAux = pd.DataFrame(np.asarray(dist), columns=list(['dist_Tbar']))            
    distTbar = pd.concat([distTbar, distAux], axis=0)

    distTbar = distTbar.reset_index(drop=True)
    data = pd.concat([data, distTbar], axis=1)
    data = data.reset_index(drop=True)
    
    
    Vx = data['VELOCITY_EW'].values
    Vy = data['VELOCITY_NS'].values
    V = (np.sqrt(np.square(Vx) + np.square(Vy)))
    angleV = calc_angle(Vx, Vy)
    Vel = np.stack((V, angleV), axis=-1)    
    Vel = pd.DataFrame(Vel, columns=list(['Velocity', 'Angle_V']))
    Vel = Vel.reset_index(drop=True)

    data = pd.concat([data, Vel], axis=1)
    data = data.reset_index(drop=True)

    Ax = data['ACCELERATION_EW'].values
    Ay = data['ACCELERATION_NS'].values
    A = (np.sqrt(np.square(Ax) + np.square(Ay)))
    angleA = calc_angle(Ax, Ay)
    Acc = np.stack((A, angleA), axis=-1)    
    Acc = pd.DataFrame(Acc, columns=list(['Acceleration', 'Angle_A']))
    Acc = Acc.reset_index(drop=True)


    data = pd.concat([data, Acc], axis=1)
    data = data.reset_index(drop=True)

    V = data[['VELOCITY_EW', 'VELOCITY_NS']].values
    W = data[['WIND_EW', 'WIND_NS']].values
    Vpv = []
    for val in range(len(V)):
        Vpv.append(ortho_projection(W[val],V[val]))


    #Vp = pd.DataFrame(Vpv, columns=list(['Velocity_p', 'Angle_p']))
    Vp = pd.DataFrame(Vpv, columns=list(['Velocity_p']))
    data = pd.concat([data, Vp], axis=1)
    data = data.reset_index(drop=True)



    data['Velocity'] = convert_knot_ms(data['Velocity'].values)

    dataset = data[['ID', 
                    'CALLSIGN', 
                    'AC_REAL_MODEL',
                    'Tbar',
                    'LATITUDE_meters', 
                    'LONGITUDE_meters', 
                    'Velocity', 
                    'Angle_V', 
                    'Acceleration', 
                    'Angle_A', 
                    'dist_Tbar',
                    'Velocity_p']]
    
    return dataset
