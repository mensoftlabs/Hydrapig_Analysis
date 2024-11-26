# Script generado desde notebook

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
    import matplotlib.gridspec as gridspec
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.cluster import KMeans
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    import warnings
    warnings.filterwarnings("ignore")



    df = pd.read_csv(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\main\df_tabla_final.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    '''# 1. Ratios de eficiencia
    df['eficiencia_comida'] = df['ganancia_peso'] / df['segundos_comedero']
    df['eficiencia_agua'] = df['ganancia_peso'] / df['segundos_bebedero']
    df['frecuencia_comida'] = df['eventos_comedero'] / df['segundos_comedero']
    df['frecuencia_bebida'] = df['eventos_bebedero'] / df['segundos_bebedero']
    df['ml_evento'] = df['mililitros'] / df['eventos_bebedero']
    df['ml_seg'] = df['mililitros'] / df['segundos_bebedero']
    df['comedero_bebedero'] = df['segundos_comedero'] / df['segundos_bebedero']
    df['interaccion_comida_bebida'] = df['segundos_comedero'] * df['segundos_bebedero']
    df['interaccion_eventos_comida'] = df['eventos_comedero'] * df['segundos_comedero']
    df
    df['mililitros_normalizado'] = (df['mililitros'] - df['mililitros'].min()) / (df['mililitros'].max() - df['mililitros'].min())
    df['mililitros_lag_1'] = df.groupby('ID')['mililitros'].shift(1)'''
    df = df.dropna()
    X = df[['eficiencia_comida',
            'eficiencia_agua','ml_evento',
            'ml_seg',
            'comedero_bebedero',
            'mililitros',
            'segundos_bebedero',
            'segundos_comedero',
            'eventos_comedero',
            'eventos_bebedero',
            'frecuencia_comida',
            'frecuencia_bebida',
            'interaccion_comida_bebida',
            'interaccion_eventos_comida']]
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y = df['ganancia_peso']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación entre variables")
    plt.show()
    mlp = MLPRegressor(max_iter=1000, random_state=42)
    param_grid = {
        'hidden_layer_sizes': [(128, 64, 32), (100, 50, 25), (64, 32, 16)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'tanh']
    }
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Mejores hiperparámetros:", grid_search.best_params_)
    best_mlp = grid_search.best_estimator_
    best_mlp.fit(X_train, y_train)
    y_pred_mlp = best_mlp.predict(X_test)
    y_train_mlp = best_mlp.predict(X_train)
    mse_test_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    mse_train_mlp = np.sqrt(mean_squared_error(y_train, y_train_mlp))
    print("RMSE Test Redes Neuronales mejoradas:", mse_test_mlp)
    print("RMSE Train Redes Neuronales mejoradas:", mse_train_mlp)
    cv_scores = cross_val_score(best_mlp, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print("MSE Promedio de la validación cruzada:", -np.mean(cv_scores))
    import joblib
    joblib.dump(best_mlp, r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\models\modelo_mlp_ganancia_peso.pkl")
    print("Modelo guardado exitosamente.")
    plt.plot(y_test.values, label='Valores Reales')
    plt.plot(y_pred_mlp, label='Predicciones MLP')
    plt.legend()
    plt.title("Comparación de predicciones")
    plt.show()
    from sklearn.inspection import permutation_importance
    result = permutation_importance(best_mlp, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    mlp_importances_df = pd.DataFrame({
        'Características': X_train.columns,
        'Importancia': result.importances_mean
    })
    mlp_importances_df = mlp_importances_df.sort_values(by='Importancia', ascending=False)
    sns.barplot(x='Importancia', y='Características', data=mlp_importances_df)
    plt.title("Importancia detallada de las características (Permutation Importance)")
    plt.figtext(0.5, -0.1, "Este gráfico muestra la importancia de las características en el modelo de red neuronal,\n"
                           "según la Permutation Importance. Indica cómo cambia la precisión cuando se altera una característica.", 
                ha="center", fontsize=10, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})
    plt.show()
    X_kmeans = df[['eficiencia_comida',
            'eficiencia_agua','ml_evento',
            'ml_seg',
            'comedero_bebedero',
            'mililitros',
            'segundos_bebedero',
            'segundos_comedero',
            'eventos_comedero',
            'eventos_bebedero',
            'frecuencia_comida',
            'frecuencia_bebida',
            'interaccion_comida_bebida',
            'interaccion_eventos_comida']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_kmeans)
    cluster_means_scaled = df.groupby('cluster').mean(numeric_only=True)
    cluster_means_scaled
    print("Media de variables desescaladas por cluster:\n", cluster_means_scaled)
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("Número de datos por cluster:\n", cluster_counts)
    import pandas as pd
    import xgboost as xgb
    from sklearn.cluster import KMeans
    import numpy as np
    df_cluster_1 = df[df['cluster'] == 1]
    min_samples = 3  # Número mínimo de muestras para continuar subdividiendo
    max_iterations = 10  # Número máximo de iteraciones para evitar bucles infinitos
    def analyze_and_cluster(data, iteration=1):
        print(f"Iteración {iteration}")
        X = data[['eficiencia_comida',
            'eficiencia_agua',
            'ml_evento',
            'ml_seg',
            'comedero_bebedero',
            'mililitros',
            'segundos_bebedero',
            'segundos_comedero',
            'eventos_comedero',
            'eventos_bebedero',
            'frecuencia_comida',
            'frecuencia_bebida',
            'interaccion_comida_bebida',
            'interaccion_eventos_comida']]
        y = data['ganancia_peso']
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xg_reg.fit(X, y)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['sub_cluster'] = kmeans.fit_predict(X)
        cluster_means = data.groupby('sub_cluster')['ganancia_peso'].mean()
        cluster_counts = data['sub_cluster'].value_counts().sort_index()
        print(f"Media de ganancia de peso por sub_cluster:\n{cluster_means}")
        print(f"Número de datos por sub_cluster:\n{cluster_counts}")
        valid_clusters = cluster_means[cluster_counts > 1].sort_values(ascending=False)
        if valid_clusters.empty:
            print("No hay sub_clusters con más de un dato")
            return None
        top_cluster = valid_clusters.idxmax()
        selected_data = data[data['sub_cluster'] == top_cluster]
        if len(selected_data) < min_samples or iteration >= max_iterations:
            print(f"Proceso terminado en iteración {iteration}")
            numeric_data = selected_data.select_dtypes(include=[np.number])
            maximos = numeric_data.max()
            minimos = numeric_data.min()
            medias = numeric_data.mean()
            df_resultado = pd.DataFrame({
                'Máximo': maximos,
                'Mínimo': minimos,
                'Media': medias
            }).T
            df_resultado = df_resultado[['ganancia_peso', 'mililitros', 'segundos_comedero', 
                                         'eventos_comedero', 'segundos_bebedero', 'eventos_bebedero', 
                                         'ml_evento', 'ml_seg']]
            return df_resultado  # Devolver el DataFrame final con los valores calculados
        return analyze_and_cluster(selected_data.drop(columns='sub_cluster'), iteration + 1)
    df_resultado_final = analyze_and_cluster(df_cluster_1)
    print("Máximos, mínimos y medias del último cluster seleccionado:\n", df_resultado_final)
    if df_resultado_final is not None:
        df_resultado_final.to_excel(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\bodyweight\max\df_ganacia_peso_max_total.xlsx")
    import pandas as pd
    import xgboost as xgb
    from sklearn.cluster import KMeans
    import numpy as np
    df_cluster_1 = df[df['cluster'] == 0]
    min_samples = 3  # Número mínimo de muestras para continuar subdividiendo
    max_iterations = 10  # Número máximo de iteraciones para evitar bucles infinitos
    def analyze_and_cluster(data, iteration=1):
        print(f"Iteración {iteration}")
        X = data[['eficiencia_comida',
            'eficiencia_agua',
            'ml_evento',
            'ml_seg',
            'comedero_bebedero',
            'mililitros',
            'segundos_bebedero',
            'segundos_comedero',
            'eventos_comedero',
            'eventos_bebedero',
            'frecuencia_comida',
            'frecuencia_bebida',
            'interaccion_comida_bebida',
            'interaccion_eventos_comida']]
        y = data['ganancia_peso']
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xg_reg.fit(X, y)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['sub_cluster'] = kmeans.fit_predict(X)
        cluster_means = data.groupby('sub_cluster')['ganancia_peso'].mean()
        cluster_counts = data['sub_cluster'].value_counts().sort_index()
        print(f"Media de ganancia de peso por sub_cluster:\n{cluster_means}")
        print(f"Número de datos por sub_cluster:\n{cluster_counts}")
        valid_clusters = cluster_means[cluster_counts > 1].sort_values()
        if valid_clusters.empty:
            print("No hay sub_clusters con más de un dato")
            return None
        lowest_cluster = valid_clusters.idxmin()
        selected_data = data[data['sub_cluster'] == lowest_cluster]
        if len(selected_data) < min_samples or iteration >= max_iterations:
            print(f"Proceso terminado en iteración {iteration}")
            numeric_data = selected_data.select_dtypes(include=[np.number])
            maximos = numeric_data.max()
            minimos = numeric_data.min()
            medias = numeric_data.mean()
            df_resultado = pd.DataFrame({
                'Máximo': maximos,
                'Mínimo': minimos,
                'Media': medias
            }).T
            df_resultado = df_resultado[['ganancia_peso', 'mililitros', 'segundos_comedero', 
                                         'eventos_comedero', 'segundos_bebedero', 'eventos_bebedero', 
                                         'ml_evento', 'ml_seg']]
            return df_resultado  # Devolver el DataFrame final con los valores calculados
        return analyze_and_cluster(selected_data.drop(columns='sub_cluster'), iteration + 1)
    df_resultado_final = analyze_and_cluster(df_cluster_1)
    print("Máximos, mínimos y medias del último cluster seleccionado:\n", df_resultado_final)
    if df_resultado_final is not None:
        df_resultado_final.to_excel(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\bodyweight\min\df_ganacia_peso_min_total.xlsx")

if __name__ == "__main__":
    main()
