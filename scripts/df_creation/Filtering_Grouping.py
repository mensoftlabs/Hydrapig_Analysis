def main():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pylab as plt
    from matplotlib.dates import date2num
    import os
    import psycopg2
    import json
    from datetime import datetime
    import math
    import warnings
    import matplotlib.gridspec as gridspec
    warnings.filterwarnings("ignore")



    conn_param_file = r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Raw\creds\farm_processdata_conn_config.json"
    with open(conn_param_file, 'r') as f:
        configdb = json.load(f)
    cnx = psycopg2.connect(**configdb)
    def select(query, cnx):
        cursor = cnx.cursor()
        try:
            cursor.execute(query)
            return(cursor.fetchall())
        except psycopg2.Error as err:
            print(str(err))
    query = """
    SELECT
        "WaterEvent".milliliters,
        "WaterEvent".seconds,
        "Chip".identifier,
        TO_CHAR("WaterEvent"."createdAt", 'YYYY-MM-DD'),
        TO_CHAR("WaterEvent"."createdAt", 'HH24:MI:SS'),
        "Device".mac,
        "Device".box
    FROM
        "WaterEvent"
    JOIN
        "Device"
    ON
        "WaterEvent"."deviceUuid" = "Device".Uuid
    JOIN
        "Chip"
    ON
        "WaterEvent"."chipUuid" = "Chip".uuid
    WHERE "Chip".identifier NOT IN ('0') AND "WaterEvent"."createdAt" > '2024-09-11'
    """
    data = select(query, cnx)
    cnx.close()

    
    conn_param_file = r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Raw\creds\farm_processdata_conn_config_comederos.json"
    with open(conn_param_file, 'r') as f:
        configdb = json.load(f)
    cnx = psycopg2.connect(**configdb)
    def select(query, cnx):
        cursor = cnx.cursor()
        try:
            cursor.execute(query)
            return(cursor.fetchall())
        except psycopg2.Error as err:
            print(str(err))
    query = """
    SELECT
    	evento.id_chip,
    	evento.boca,
    	evento.segundos,
    	evento.creado,
    	dispositivo.numero
    FROM
    	evento
    JOIN dispositivo ON evento."dispositivoId" = dispositivo.id
    WHERE evento.creado > '2024-09-11';
    """
    data2 = select(query, cnx)
    cnx.close()
    caudalimetro_df = pd.DataFrame(data, columns=["milliliters",
                                    "seconds",
                                    "chip",
                                    "fecha",
                                    "hora",
                                    "mac",
                                    "box"])
    comedero_df = pd.DataFrame(data2, columns=["chip",
                                    "boca",
                                    "segundos",
                                    "fecha",
                                    "numero"])
    comedero_df
    # **FILTRADO DE CAUDALIMETRO**
    # In[3]:
    # Convertir 'fecha' a datetime
    caudalimetro_df['fecha'] = pd.to_datetime(caudalimetro_df['fecha'], format='%Y-%m-%d')# Crear la columna 'timestamp' combinando 'fecha' y 'hora_completa'
    caudalimetro_df['timestamp'] = pd.to_datetime(caudalimetro_df['fecha'].astype(str) + ' ' + caudalimetro_df['hora'])
    caudalimetro_df['timestamp'] = caudalimetro_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Filtramos por columnas relevantes, por ejemplo: 'lechon_id', 'timestamp', 'drink_duration', 'drink_volume'
    caudalimetro_df = caudalimetro_df[['chip','box', 'timestamp', 'seconds', 'milliliters']]
    # Opcional: Eliminar filas con timestamps inválidos (por ejemplo, si hay algún error en los datos)
    caudalimetro_df = caudalimetro_df.dropna(subset=['timestamp'])
    # Convertir 'crotal' a string
    caudalimetro_df['chip'] = caudalimetro_df['chip'].astype(str)
    # Función para obtener los últimos 4 dígitos y ajustar si el primer dígito es 0
    def obtener_id_cerdo(crotal):
        ultimos_cuatro = crotal[-4:]
        if ultimos_cuatro[0] == '0':
            return ultimos_cuatro[-3:]
        return ultimos_cuatro
    # Crear la columna 'ID_Cerdo'
    caudalimetro_df['ID'] = caudalimetro_df['chip'].apply(obtener_id_cerdo)
    # Reordenar las columnas: 'timestamp', 'ID_Cerdo', 'corral', 'segundos', 'mililitros'
    caudalimetro_df = caudalimetro_df[['timestamp', 'ID', 'box', 'seconds', 'milliliters']]
    caudalimetro_df['segundos'] = caudalimetro_df['seconds']
    caudalimetro_df['mililitros'] = caudalimetro_df['milliliters']
    caudalimetro_df = caudalimetro_df.drop(columns=['seconds','milliliters'])
    caudalimetro_df = caudalimetro_df.sort_values(by=['timestamp'])
    import pandas as pd
    # Función que agrupa eventos con menos de 3 segundos de diferencia y suma los valores de 'segundos'
    def agrupar_timestamps(df):
        # Ordenamos el DataFrame por 'ID' y 'timestamp' para facilitar el procesamiento
        df = df.sort_values(by=['ID', 'timestamp'])
        # Creamos una lista para almacenar los índices a eliminar
        indices_a_eliminar = []
        # Iteramos sobre el DataFrame agrupado por 'ID'
        for id_cerdo, grupo in df.groupby('ID'):
            # Iteramos sobre las filas del grupo con la función iterrows()
            i = 0
            while i < len(grupo) - 1:
                # Calculamos la diferencia de tiempo entre eventos consecutivos
                diff = (grupo.iloc[i+1]['timestamp'] - grupo.iloc[i]['timestamp']).total_seconds()
                # Si la diferencia es menor de 3 segundos, sumamos 'segundos' y marcamos para eliminar
                if diff < 10:
                    df.loc[grupo.index[i], 'segundos'] += df.loc[grupo.index[i+1], 'segundos']
                    df.loc[grupo.index[i], 'mililitros'] += df.loc[grupo.index[i+1], 'mililitros']
                    indices_a_eliminar.append(grupo.index[i+1])
                i += 1
        # Eliminamos los índices de eventos redundantes
        df = df.drop(indices_a_eliminar)
        return df
    # Aplicar la función al DataFrame comedero_df
    caudalimetro_df['timestamp'] = pd.to_datetime(caudalimetro_df['timestamp'])  # Asegurar que los timestamps sean del tipo datetime
    caudalimetro_df_filtrado = agrupar_timestamps(caudalimetro_df)
    # Eliminar filas con 'mililitros' == 0 y 'segundos' < 3
    caudalimetro_df_filtrado = caudalimetro_df_filtrado[~((caudalimetro_df_filtrado['mililitros'] == 0) & (caudalimetro_df_filtrado['segundos'] < 3))]
    # Encontrar el valor máximo de 'mililitros' y el valor de 'segundos' y 'timestamp' asociado
    max_mililitros = caudalimetro_df_filtrado['mililitros'].max()
    fila_max_mililitros = caudalimetro_df_filtrado.loc[caudalimetro_df_filtrado['mililitros'] == max_mililitros]
    segundos_asociados = fila_max_mililitros['segundos'].values[0]
    timestamp_asociado = fila_max_mililitros['timestamp'].values[0]
    print(f"El valor máximo de 'mililitros' es {max_mililitros}, el valor de 'segundos' asociado es {segundos_asociados}, y corresponde al día {timestamp_asociado}.")
    # Ver el resultado
    caudalimetro_df_filtrado.sort_values(by=['timestamp'])
    caudalimetro_df_filtrado = caudalimetro_df_filtrado[~((caudalimetro_df_filtrado['mililitros'] == 0) & (caudalimetro_df_filtrado['segundos'] > 30))]
    caudalimetro_df_filtrado = caudalimetro_df_filtrado[~((caudalimetro_df_filtrado['mililitros'] == 0) & (caudalimetro_df_filtrado['segundos'] == 0))]
    # **FILTRADO COMEDERO**
    # In[4]:
    # Convertir 'fecha' a datetime
    comedero_df['timestamp'] = comedero_df['fecha'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Filtramos por columnas relevantes, por ejemplo: 'lechon_id', 'timestamp', 'drink_duration', 'drink_volume'
    comedero_df = comedero_df[['chip','numero','boca','timestamp','segundos']]
    # Opcional: Eliminar filas con timestamps inválidos (por ejemplo, si hay algún error en los datos)
    comedero_df = comedero_df.dropna(subset=['timestamp'])
    # Filtramos por columnas relevantes
    comedero_df = comedero_df[['chip', 'numero', 'boca', 'timestamp','segundos']]
    # Opcional: Eliminar filas con timestamps inválidos
    comedero_df = comedero_df.dropna(subset=['timestamp'])
    # Convertir 'crotal' a string
    comedero_df['chip'] = comedero_df['chip'].astype(str)
    # Función para obtener los últimos 4 dígitos y ajustar si el primer dígito es 0
    def obtener_id_cerdo(crotal):
        ultimos_cuatro = crotal[-4:]
        if ultimos_cuatro[0] == '0':
            return ultimos_cuatro[-3:]
        return ultimos_cuatro
    # Crear la columna 'ID_Cerdo'
    comedero_df['ID'] = comedero_df['chip'].apply(obtener_id_cerdo)
    # Reordenar las columnas: 'timestamp', 'ID_Cerdo', 'corral', 'segundos', 'mililitros'
    comedero_df = comedero_df[['timestamp', 'ID', 'numero', 'boca', 'segundos']]
    comedero_df['box'] = comedero_df['numero']
    comedero_df = comedero_df.drop(columns=['numero'])
    comedero_df = comedero_df[['timestamp', 'ID', 'box', 'boca', 'segundos']]
    comedero_df = comedero_df.sort_values(by=['timestamp'])
    comedero_df['segundos'] = pd.to_numeric(comedero_df['segundos'], errors='coerce')
    comedero_df = comedero_df[comedero_df['segundos'] > 3]
    import pandas as pd
    # Función que agrupa eventos con menos de 3 segundos de diferencia y suma los valores de 'segundos'
    def agrupar_timestamps(df):
        # Ordenamos el DataFrame por 'ID' y 'timestamp' para facilitar el procesamiento
        df = df.sort_values(by=['ID', 'timestamp'])
        # Creamos una lista para almacenar los índices a eliminar
        indices_a_eliminar = []
        # Iteramos sobre el DataFrame agrupado por 'ID'
        for id_cerdo, grupo in df.groupby('ID'):
            # Iteramos sobre las filas del grupo con la función iterrows()
            i = 0
            while i < len(grupo) - 1:
                # Calculamos la diferencia de tiempo entre eventos consecutivos
                diff = (grupo.iloc[i+1]['timestamp'] - grupo.iloc[i]['timestamp']).total_seconds()
                # Si la diferencia es menor de 3 segundos, sumamos 'segundos' y marcamos para eliminar
                if diff < 10:
                    df.loc[grupo.index[i], 'segundos'] += df.loc[grupo.index[i+1], 'segundos']
                    indices_a_eliminar.append(grupo.index[i+1])
                i += 1
        # Eliminamos los índices de eventos redundantes
        df = df.drop(indices_a_eliminar)
        return df
    # Aplicar la función al DataFrame comedero_df
    comedero_df['timestamp'] = pd.to_datetime(comedero_df['timestamp'])  # Asegurar que los timestamps sean del tipo datetime
    comedero_df_filtrado = agrupar_timestamps(comedero_df)
    # Ver el resultado
    comedero_df_filtrado.sort_values(by=['timestamp'])
    comedero_df_filtrado = comedero_df_filtrado[~((comedero_df_filtrado['segundos'] > 90))]
    # In[5]:
    comedero_df_filtrado
    # In[6]:
    caudalimetro_df_filtrado
    # In[7]:
    # Redondear los timestamps a la hora más cercana
    comedero_df_filtrado['timestamp'] = comedero_df_filtrado['timestamp'].dt.floor('H')
    caudalimetro_df_filtrado['timestamp'] = caudalimetro_df_filtrado['timestamp'].dt.floor('H')
    # Unir los dataframes por ID y hora
    df_final = pd.merge(comedero_df_filtrado, caudalimetro_df_filtrado, on=['ID', 'timestamp'], how='outer')
    df_final['segundos_comiendo'] = df_final['segundos_x']
    df_final['segundos_bebiendo'] = df_final['segundos_y']
    df_final['box_comedero'] = df_final['box_x']
    df_final['box_bebedero'] = df_final['box_y']
    # Mostrar el resultado
    df_final = df_final.drop(columns=['segundos_x','segundos_y','box_x','box_y'])
    import pandas as pd
    # Convertir la columna 'timestamp' a tipo datetime
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
    # Agrupar por ID y día, y sumar las variables de interés
    df_final = df_final.groupby(['ID', df_final['timestamp'].dt.date]).agg({
        'mililitros': ['sum'],
        'segundos_comiendo': ['sum', 'count'],
        'segundos_bebiendo': ['sum', 'count']
    }).reset_index()
    # Convertir la columna 'timestamp' agrupada a datetime
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
    df_final = df_final.drop(0)
    # Filtrar los datos hasta 2024-10-07
    df_final = df_final[df_final['timestamp'] <= pd.to_datetime('2024-10-22')]
    df_final.columns = ['ID','timestamp','mililitros','segundos_comedero','eventos_comedero','segundos_bebedero','eventos_bebedero']
    # Definir los IDs a eliminar
    ids_a_eliminar = ['5694', '8096', '8100','440','9787']
    # Filtrar el DataFrame para mantener solo las filas que no contienen los IDs especificados
    df_final = df_final[~df_final['ID'].isin(ids_a_eliminar)]
    # Supongamos que df_final ya está definido y contiene los datos
    # Definir el rango de fechas (solo fechas, sin hora)
    start_date = df_final['timestamp'].min().date()
    end_date = df_final['timestamp'].max().date()
    all_timestamps = pd.date_range(start=start_date, end=end_date).date  # Solo fechas
    # Crear un DataFrame con todos los IDs y todos los timestamps
    all_ids = df_final['ID'].unique()
    full_combinations = pd.MultiIndex.from_product([all_ids, all_timestamps], names=['ID', 'timestamp']).to_frame(index=False)
    # Asegurarse de que 'timestamp' en df_final sea de tipo fecha
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp']).dt.date  # Convertir a solo fecha
    # Hacer un merge con df_final para ver cuáles son los registros que faltan
    missing_data = full_combinations.merge(df_final[['ID', 'timestamp']], on=['ID', 'timestamp'], how='left', indicator=True)
    # Filtrar para obtener solo las filas donde el timestamp no está presente en df_final
    missing_timestamps = missing_data[missing_data['_merge'] == 'left_only']
    # Agrupar por ID y listar los timestamps que faltan
    missing_summary = missing_timestamps.groupby('ID')['timestamp'].apply(list).reset_index()
    # Renombrar las columnas
    missing_summary.columns = ['ID', 'missing_timestamps']



    pesajes_df = pd.read_csv(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Raw\Pesajes_Total.csv")
    pesajes_df['ID'] = pesajes_df['ID_Cerdo']
    pesajes_df = pesajes_df[['ID', 'BW0', 'BW1', 'BW2', 'BW3', 'BW4','BW5','BW6']]
    pesajes_df['BW0'] = pesajes_df['BW0'].str.replace(',', '.').astype(float)
    pesajes_df['BW1'] = pesajes_df['BW1'].str.replace(',', '.').astype(float)
    pesajes_df['BW2'] = pesajes_df['BW2'].str.replace(',', '.').astype(float)
    pesajes_df['BW3'] = pesajes_df['BW3'].str.replace(',', '.').astype(float)
    pesajes_df['BW4'] = pesajes_df['BW4'].str.replace(',', '.').astype(float)
    pesajes_df['BW5'] = pesajes_df['BW5'].str.replace(',', '.').astype(float)
    pesajes_df['BW6'] = pesajes_df['BW6'].str.replace(',', '.').astype(float)
    pesajes_df = pesajes_df.sort_values(by=['ID'])
    pesajes_df["BW2"].fillna(0, inplace=True)
    pesajes_df["BW3"].fillna(0, inplace=True)
    pesajes_df["BW4"].fillna(0, inplace=True)
    pesajes_df["BW5"].fillna(0, inplace=True)
    pesajes_df["BW6"].fillna(0, inplace=True)
    # Definir las fechas asociadas a cada columna BW
    dates = ["2024-09-12", "2024-09-16", "2024-09-23", "2024-09-30", "2024-10-07","2024-10-14","2024-10-20"]
    # Transformar el DataFrame de formato ancho a formato largo con 'melt'
    df_melted = pd.melt(pesajes_df, id_vars=["ID"], value_vars=["BW0", "BW1", "BW2", "BW3", "BW4", "BW5", "BW6"], 
                        var_name="timestamp", value_name="peso")
    # Reemplazar los valores de BW0, BW1, etc. por las fechas correspondientes
    df_melted["timestamp"] = df_melted["timestamp"].map({
        "BW0": dates[0], 
        "BW1": dates[1], 
        "BW2": dates[2], 
        "BW3": dates[3], 
        "BW4": dates[4],
        "BW5": dates[5],
        "BW6": dates[6]
    })
    df_melted['timestamp'] = pd.to_datetime(df_melted['timestamp'])
    # Rellenar los valores NaN en la columna 'peso' con 0
    df_melted["peso"].fillna(0, inplace=True)
    # Generar un rango de fechas desde el 12 de septiembre al 7 de octubre
    full_date_range = pd.date_range(start="2024-09-12", end="2024-10-21")
    # Crear un DataFrame auxiliar con todas las combinaciones de IDs y fechas
    unique_ids = df_melted["ID"].unique()  # Obtener todos los IDs únicos
    expanded_df = pd.MultiIndex.from_product([unique_ids, full_date_range], names=["ID", "timestamp"]).to_frame(index=False)
    # Fusionar el DataFrame expandido con los datos de pesaje originales
    df_full = pd.merge(expanded_df, df_melted, on=["ID", "timestamp"], how="left")
    # Rellenar los valores NaN en la columna 'peso' con 0 para los días sin pesaje
    df_full["peso"].fillna(0, inplace=True)
    import pandas as pd
    # Supongamos que df_full y missing_summary ya están definidos
    # Asegurarse de que 'ID' sea del mismo tipo en ambos DataFrames
    df_full['ID'] = df_full['ID'].astype(str).str.strip()  # Convertir a str y eliminar espacios
    missing_summary['ID'] = missing_summary['ID'].astype(str).str.strip()  # Hacer lo mismo en missing_summary
    # Crear un DataFrame a partir de missing_summary con las combinaciones a eliminar
    conditions_to_remove = missing_summary.explode('missing_timestamps').rename(columns={'missing_timestamps': 'timestamp'})
    conditions_to_remove['timestamp'] = pd.to_datetime(conditions_to_remove['timestamp'])  # Asegurarse de que sea datetime
    # Filtrar df_full para eliminar las filas que coinciden con las condiciones
    to_remove = set(zip(conditions_to_remove['ID'], conditions_to_remove['timestamp']))
    # Filtrar df_full para mantener solo las filas que NO están en to_remove
    df_full_filtered = df_full[~df_full.set_index(['ID', 'timestamp']).index.isin(to_remove)]
    # In[10]:
    # Supongamos que df_full_filtered y df_final ya están definidos
    # Asegurarse de que el tipo de dato de 'timestamp' sea datetime
    df_full_filtered['timestamp'] = pd.to_datetime(df_full_filtered['timestamp'])
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
    # Realizar la unión de los DataFrames
    merged_df = pd.merge(df_full_filtered, df_final, on=['ID', 'timestamp'], how='outer', suffixes=('_full', '_final'))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    # 1. Cargar los datos
    df = merged_df
    # Asegurarse de que las fechas están en el formato adecuado
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['ID', 'timestamp'])
    # Reemplazar 0 por NaN en la columna 'peso' para identificar los días sin mediciones reales
    df['peso'] = df['peso'].replace(0, np.nan)
    # 2. Crear características de diferencias diarias
    df['delta_mililitros'] = df.groupby('ID')['mililitros'].diff().fillna(0)
    df['delta_comiendo'] = df.groupby('ID')['segundos_comedero'].diff().fillna(0)
    df['delta_bebiendo'] = df.groupby('ID')['segundos_bebedero'].diff().fillna(0)
    # Calcular la diferencia en el número de días entre mediciones
    df['days_diff'] = df.groupby('ID')['timestamp'].diff().dt.days.fillna(1)
    # Función para dividir los datos en intervalos definidos por las fechas con peso real
    def split_into_intervals(group):
        # Identificar las filas con valores de peso no nulos (mediciones reales)
        real_weight_indices = group[group['peso'].notna()].index
        # Generar intervalos entre las fechas con peso real
        intervals = []
        for i in range(len(real_weight_indices) - 1):
            start_idx = real_weight_indices[i]
            end_idx = real_weight_indices[i + 1]
            interval = group.loc[start_idx:end_idx]
            intervals.append(interval)
        return intervals
    # 3. Aplicar la función para cada ID
    df_intervals = []
    for _, group in df.groupby('ID'):
        df_intervals.extend(split_into_intervals(group))
    # 4. Crear el dataset de entrenamiento basado en los intervalos
    train_data = []
    for interval in df_intervals:
        # Calcular el incremento total de peso en este intervalo
        total_weight_change = interval['peso'].iloc[-1] - interval['peso'].iloc[0]
        # Solo incluimos los días sin medición real (días intermedios)
        for i in range(1, len(interval) - 1):  # Evitar el primer y último día, que ya tienen peso real
            delta_mililitros = interval['delta_mililitros'].iloc[i]
            delta_comiendo = interval['delta_comiendo'].iloc[i]
            delta_bebiendo = interval['delta_bebiendo'].iloc[i]
            days_diff = interval['days_diff'].iloc[i]
            train_data.append({
                'ID': interval['ID'].iloc[i],
                'timestamp': interval['timestamp'].iloc[i],
                'delta_mililitros': delta_mililitros,
                'delta_comiendo': delta_comiendo,
                'delta_bebiendo': delta_bebiendo,
                'days_diff': days_diff,
                'weight_change_ratio': np.nan  # Este es el objetivo a predecir
            })
    df_train = pd.DataFrame(train_data)
    # 5. Normalizar el incremento total entre los días intermedios
    for interval in df_intervals:
        total_weight_change = interval['peso'].iloc[-1] - interval['peso'].iloc[0]
        # Dividir el incremento total de peso proporcionalmente a los días intermedios
        interval_weight_change_per_day = total_weight_change / (len(interval) - 1)
        # Asignar un cambio de peso proporcional para cada día intermedio
        df_train.loc[df_train['ID'] == interval['ID'].iloc[0], 'weight_change_ratio'] = interval_weight_change_per_day
    # 6. Entrenar el modelo
    X = df_train[['delta_mililitros', 'delta_comiendo', 'delta_bebiendo', 'days_diff']]
    y = df_train['weight_change_ratio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Modelo Random Forest
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    # 7. Predicción para los días sin mediciones reales en cada intervalo
    df_train['ganancia_peso'] = rf_reg.predict(X)
    # 8. Volver a unir las predicciones a cada intervalo original
    df_train_with_predictions = df_train[['ID', 'timestamp', 'ganancia_peso']]
    # Función para aplicar las predicciones y ajustar progresivamente los pesos
    def apply_weight_predictions(interval, predictions_df):
        # Unir las predicciones con el intervalo original
        interval = pd.merge(interval, predictions_df, on=['ID', 'timestamp'], how='left')
        # Identificar el peso inicial y el peso final (real) del intervalo
        initial_weight = interval['peso'].iloc[0]
        final_weight = interval['peso'].iloc[-1]
        # Calcular la diferencia total de peso que debe distribuirse entre los días intermedios
        total_weight_change = final_weight - initial_weight
        # Filtrar los días intermedios (sin peso real)
        intermedios = interval.iloc[1:-1]
        # Ajustar las predicciones para que la suma de los cambios de peso sea igual al total_weight_change
        predicted_total_change = intermedios['ganancia_peso'].sum()
        # Si el cambio total predicho es distinto al real, ajustamos las predicciones
        if predicted_total_change != 0:
            adjustment_factor = total_weight_change / predicted_total_change
            intermedios['ganancia_peso'] *= adjustment_factor
        # Asegurar que todos los valores en 'ganancia_peso' sean distintos de cero
        intermedios['ganancia_peso'] = intermedios['ganancia_peso'].replace(0, np.random.uniform(0.01, 0.05))
        # Aplicar las predicciones ajustadas y evitar que el peso predicho sea igual al anterior
        for i in range(1, len(interval) - 1):
            interval['peso'].iloc[i] = interval['peso'].iloc[i - 1] + intermedios['ganancia_peso'].iloc[i - 1]
            # Asegurar que el peso predicho no sea igual al día anterior
            if interval['peso'].iloc[i] == interval['peso'].iloc[i - 1]:
                interval['peso'].iloc[i] += np.random.uniform(0.01, 0.05)  # Añadir una pequeña diferencia
        return interval
    # 10. Rellenar los pesos faltantes en cada intervalo respetando los límites y asegurando progresión
    df_filled_intervals = []
    for interval in df_intervals:
        df_filled_intervals.append(apply_weight_predictions(interval, df_train_with_predictions))
    # Combinar todos los intervalos completados en un solo DataFrame
    df_filled = pd.concat(df_filled_intervals)
    # 11. Guardar el resultado final
    df_filled = df_filled[['ID', 'timestamp', 'peso', 'mililitros', 'segundos_comedero','eventos_comedero', 'segundos_bebedero','eventos_bebedero','ganancia_peso']]
    df_filled['ganancia_peso'] = df_filled['ganancia_peso'].fillna(0)
    # Asegurar que no haya valores de ganancia_peso igual a cero
    df_filled['ganancia_peso'] = df_filled['ganancia_peso'].replace(0, np.random.uniform(0.1, 0.3))
    # Mostrar el DataFrame final sin ganancia_peso igual a cero
    # In[12]:
    count = df_filled[(df_filled['ganancia_peso'] == 0) ].shape[0]
    print(f"El número de filas con 0 en 'ganancia_peso': {count}")
    # In[13]:
    # DataFrame eliminando las filas donde 'mililitros' y 'segundos_bebiendo' son ambos 0
    df_filtered = df_filled[~((df_filled['mililitros'] == 0) & (df_filled['segundos_bebedero'] == 0))]
    df_filtered = df_filtered[df_filtered['ganancia_peso'] <= 1]
    df_filtered = df_filtered[df_filtered['ganancia_peso'] >= -0.5]
    df_filtered = df_filtered[~df_filtered['ID'].isin(['440','9787','9761','9770','434'])]
    # Obtener los 10 valores máximos de una variable y sus respectivos IDs
    top_10_max_values = df_filtered[['ID', 'ganancia_peso']].nlargest(10, 'ganancia_peso')
    print(top_10_max_values)
    # Lista de timestamps que deseas eliminar
    timestamps_a_eliminar = ['2024-09-12', '2024-09-16', '2024-09-23', '2024-09-30', '2024-10-07','2024-10-14','2024-10-20']
    # Convertir los timestamps a formato datetime si no lo están
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
    # Filtrar el DataFrame para eliminar las filas con esos timestamps
    df_filtered = df_filtered[~df_filtered['timestamp'].isin(timestamps_a_eliminar)]
    # Mostrar los DataFrames
    # In[14]:
    df_filtered = df_filtered[(df_filtered['segundos_bebedero'] <= 110) & 
                            (df_filtered['segundos_comedero'] <= 350) & 
                            (df_filtered['mililitros'] <= 400)&
                            (df_filtered['eventos_comedero'] <= 35)&
                            (df_filtered['eventos_bebedero'] <= 35)]
    # 1. Ratios de eficiencia
    df_filtered['eficiencia_comida'] = df_filtered['ganancia_peso'] / df_filtered['segundos_comedero']
    df_filtered['eficiencia_agua'] = df_filtered['ganancia_peso'] / df_filtered['segundos_bebedero']
    df_filtered['frecuencia_comida'] = df_filtered['eventos_comedero'] / df_filtered['segundos_comedero']
    df_filtered['frecuencia_bebida'] = df_filtered['eventos_bebedero'] / df_filtered['segundos_bebedero']
    df_filtered['ml_evento'] = df_filtered['mililitros'] / df_filtered['eventos_bebedero']
    df_filtered['ml_seg'] = df_filtered['mililitros'] / df_filtered['segundos_bebedero']
    df_filtered['comedero_bebedero'] = df_filtered['segundos_comedero'] / df_filtered['segundos_bebedero']
    # 3. Interacciones
    df_filtered['interaccion_comida_bebida'] = df_filtered['segundos_comedero'] * df_filtered['segundos_bebedero']
    df_filtered['interaccion_eventos_comida'] = df_filtered['eventos_comedero'] * df_filtered['segundos_comedero']
    # Reemplazar valores infinitos en el DataFrame con cero
    df_filtered.replace([float('inf'), -float('inf')], 0, inplace=True)
    df_filtered.to_csv(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\main\df_tabla_final.csv", index=False)
    # In[16]:
    # Definir los intervalos de fechas específicos para septiembre 2024
    intervalos_fechas = {
        '12-15': ('2024-09-12', '2024-09-15'),
        '16-22': ('2024-09-16', '2024-09-22'),
        '23-29': ('2024-09-23', '2024-09-29'),
        '30-6': ('2024-09-30', '2024-10-06'),
        '7-13': ('2024-10-07', '2024-10-13'),
        '14-20': ('2024-10-14', '2024-10-20')
    }
    # Filtrar los datos en base a los intervalos de fechas específicos
    dfs_fechas = {}
    for nombre, (inicio, fin) in intervalos_fechas.items():
        # Convertir fechas de inicio y fin a formato datetime
        fecha_inicio = pd.to_datetime(inicio)
        fecha_fin = pd.to_datetime(fin)
        # Filtrar el DataFrame por el rango de fechas
        dfs_fechas[nombre] = df_filtered[(df_filtered['timestamp'] >= fecha_inicio) & (df_filtered['timestamp'] <= fecha_fin)]
    # Crear la carpeta si no existe
    output_folder = r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\main"
    os.makedirs(output_folder, exist_ok=True)
    # Guardar cada DataFrame de intervalo en un archivo Excel en la carpeta especificada
    for nombre, df_intervalo in dfs_fechas.items():
        filename = os.path.join(output_folder, f"df_datos_{nombre.replace('-', '_')}.xlsx")
        df_intervalo.to_excel(filename, index=False)
    # In[ ]:

if __name__ == "__main__":
    main()
