from .informalmodelinterface import InformalModelInterface
from IPython.display import display
from pandas import DataFrame
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class ClassifierModel (InformalModelInterface):    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    @property
    def IQ_LOWER_RANGE(self) -> float:
        return self._IQ_LOWER_RANGE
    
    @IQ_LOWER_RANGE.setter
    def IQ_LOWER_RANGE(self, value):
        self._IQ_LOWER_RANGE = value
    

    def __init__(self, versionLabel: str, randomSeed = 42, asTesting = False):
        self._model = None
        self._IQ_LOWER_RANGE = 1.5
        self.RANDOM_STATE_SEED = randomSeed
        self.USE_AS_TEST = asTesting


        self.MODEL_DUMP_PATH = 'dump/'
        self.BASE_DUMP_DIRECTORY = f"{self.MODEL_DUMP_PATH}{"test/" if self.USE_AS_TEST else ""}"

        # Definir los nombres de los archivos de los modelos
        
        self.knn_model_filename = f"{self.BASE_DUMP_DIRECTORY}knn_model_{versionLabel}.pkl"
        self.rf_model_filename = f"{self.BASE_DUMP_DIRECTORY}rf_model_{versionLabel}.pkl"
        self.label_encoder_filename = f"{self.BASE_DUMP_DIRECTORY}label_encoder_{versionLabel}.pkl"
        self.comparison_filename = f"{self.BASE_DUMP_DIRECTORY}model_comparison_{versionLabel}.txt"

    def cleanAndPreprocessData(self, X: DataFrame, y: DataFrame):
        # Obtener cantidad de outliers
        # Calcular Q1 y Q3 para todas las columnas
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)

        # Calcular el IQR para cada columna
        IQR = Q3 - Q1

        # Identificar outliers
        outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))

        # Para el caso de Age y Heart, no se eliminaran los outliers, 
        # serán reemplazados por la media
        X.loc[outliers['HeartRate'], 'HeartRate'] = int(X['HeartRate'].mean())
        X.loc[outliers['Age'], 'Age'] = int(X['Age'].mean())

        # Crear un nuevo dataframe para almacenar las variables independientes
        # transformadas
        XT = pd.DataFrame()

        # Aplicar la transformación inversa se añade una pequeña constante para evitar división por 0 

        XT['Age'] = np.sqrt(X['Age'])
        XT['SystolicBP']=np.sqrt(X['SystolicBP'])
        XT['DiastolicBP']=np.sqrt(X['DiastolicBP'])

        XT['HeartRate'] = 1 / (X['HeartRate'] + 1e-9)
        XT['BodyTemp']  = 1 / (X['BodyTemp'] + 1e-9)
        XT['BS'] = 1 / (X['BS'] + 1e-9)

        # Normalizar datos 
        normalizer = MinMaxScaler()
        # Aplicar la normalización 
        XT_normalized = pd.DataFrame(normalizer.fit_transform(XT), columns=XT.columns)

        # Codificar la variable target, debido a que las etiquetas son high risk, mid risk y low risk.
        label_encoder = LabelEncoder()
        yt = label_encoder.fit_transform(y)

        return XT, yt


    def splitTrainTestVal(self, X: DataFrame, y: DataFrame, trainSize: float = 0.8, testSize: float = 0.15):
        dataSize = trainSize + testSize
        if (dataSize > 1):
            raise Exception("Sum of sizes shouldn't be higher than 1")
        
        
        valSize = 1 - dataSize
        
        from sklearn.model_selection import train_test_split
        

        """
            TODO: This needs to be completed with the right second split formula
        """
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=trainSize, shuffle=True, random_state=self.RANDOM_STATE_SEED)
        Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.1764705882352941, random_state=self.RANDOM_STATE_SEED) # 0.1764705882352941 x 0.85 = 0.15

        print(Xtrain.shape, ytrain.shape)
        print(Xval.shape, yval.shape)
        print(Xtest.shape, ytest.shape)

        return (Xtrain, ytrain, Xtest, ytest, Xval, yval)

        # Peso de las clases de acuerdo al porcentaje de los datos encontrados en EDA
        # con la finalidad de compensar el desbalanceo de datos
        # el 27% de los datos corresponde a la clase "high risk:0" 
        # el 33% a la clase "mid risk:2" 
        # el 40% a la clase "low risk:1"
        # Se usará el inverso de la proporción para definir los pesos de las clases.

        class_weight = {0: 3.70, 1: 3.03, 2: 2.5}

        # Definir el modelo 
        random_forest_model = RandomForestClassifier(random_state=42,class_weight=class_weight)


        # Definir el espacio de hiperparámetros para la búsqueda
        param_grid = {
            'n_estimators': [5, 10, 20, 50],             # Número de árboles en el bosque
            'max_depth': [3, 5, 10],                     # Profundidad máxima de los árboles
            'min_samples_split': [3, 5, 10, 20],         # Número mínimo de muestras para dividir un nodo
            'min_samples_leaf': [1, 2, 4],               # Número mínimo de muestras en una hoja
            'bootstrap': [True, False]                   # Usar o no muestreo con reemplazo
        }

        # Inicializar el RandomizedSearchCV
        grid_search = GridSearchCV(estimator=random_forest_model, 
                                        param_grid=param_grid, 
                                        scoring='f1_weighted',       
                                        cv=3,                        
                                        verbose=2, 
                                        n_jobs=-1)

        # Ajustar el modelo usando la búsqueda de hiperparámetros
        grid_search.fit(X_train, y_train)

        # Mostrar los mejores parámetros
        self._model = grid_search.best_estimator_

        return self._model
    
    def predict(self, X):
        if self._model is not None:
            return self._model.predict(X)
        raise Exception("Couldn't find a model")

    def getOutliers(self, X: DataFrame):
        # Calcular Q1 y Q3 para todas las columnas
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)

        # Calcular el IQR para cada columna
        IQR = Q3 - Q1

        # Identificar outliers
        outliers = (X < (Q1 - self.IQ_LOWER_RANGE * IQR)) | (X > (Q3 + self.IQ_LOWER_RANGE * IQR))

        return outliers

    def outlierHandler(self, X: DataFrame):
        # Crear una copia del dataset para no modificar el original
        X_copy = X.copy()

        
        # Obtener los outliers del dataset
        outliers = self.getOutliers(X_copy)

        for index in X_copy.columns:
            X_copy.loc[outliers[index], index] = int(X_copy[index].mean())

        # Devolver el nuevo dataset
        return X_copy

    @staticmethod
    def applyTransformations(X: DataFrame):
        # Transformaciónes
        XT = pd.DataFrame()


        XT['Age'] = np.log(X['Age'] + 1)

        XT['SystolicBP'], _ = stats.yeojohnson(X['SystolicBP'])
        XT['DiastolicBP'], _ = stats.yeojohnson(X['DiastolicBP'])

        # Logarítmica para BS y BodyTemp
        XT['BS'] = np.log(X['BS'] + 1e-9)


        XT['BodyTemp'] = np.sqrt(X['BodyTemp'])


        # HeartRate no necesita transformación
        XT['HeartRate'] = X['HeartRate']

        return XT

    @staticmethod
    def normalizeData(X):
        """
        Normaliza los datos de un DataFrame usando MinMaxScaler.
        
        Args:
            X (pd.DataFrame): El DataFrame que contiene los datos a normalizar.

        Returns:
            pd.DataFrame: Un nuevo DataFrame con los datos normalizados.
        """
        normalizer = MinMaxScaler()
        normalized_data = pd.DataFrame(normalizer.fit_transform(X), columns=X.columns)
        return normalized_data

    def dumpLabelEncoder(self, labelEncoder):
        joblib.dump(labelEncoder, self.label_encoder_filename)
        print(f"LabelEncoder guardado como {self.label_encoder_filename}.")

    def labelEncoding(self, Y, saveEncoder = False):
        # Codificar la variable target, debido a que las etiquetas son high risk, mid risk y low risk.
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y.values.ravel())

        if (saveEncoder):
            self.dumpLabelEncoder(label_encoder)

        return Y_encoded, label_encoder

    def trainTestSplit(X,Y,test_size = 0.2):    
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        return X_train, X_test, Y_train, Y_test

    def datasetSummary(self, X, Y):
        """
        Función que une X (features) e Y (target), realiza los cálculos de outliers, skewness y kurtosis
        para las variables numéricas, y genera una tabla consolidada con esta información.

        Parámetros:
        X -- DataFrame con las variables independientes (features).
        Y -- Serie con la variable dependiente (target).
        
        Retorna:
        final_summary_df -- DataFrame con la información de las columnas unidas de X y Y,
                            incluyendo outliers, skewness y kurtosis para las variables numéricas.
        """
        # 1. Unir los datasets X y Y en uno solo (asumimos que Y es una Serie)
        combined_df = pd.concat([X, Y], axis=1)

        # 2. Filtrar solo las columnas numéricas de X para hacer los cálculos
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        # 3. Calcular Q1 y Q3 para las columnas numéricas
        Q1 = X[numeric_columns].quantile(0.25)
        Q3 = X[numeric_columns].quantile(0.75)

        # 4. Calcular el IQR para las columnas numéricas
        IQR = Q3 - Q1

        # 5. Identificar outliers solo en las columnas numéricas
        outliers = (X[numeric_columns] < (Q1 - 1.5 * IQR)) | (X[numeric_columns] > (Q3 + 1.5 * IQR))

        # 6. Contar cuántos outliers existen en cada columna numérica
        outliers_count = outliers.sum()

        # 7. Calcular skewness y kurtosis solo para las columnas numéricas
        skewness = X[numeric_columns].skew()
        kurtosis = X[numeric_columns].kurt()

        # 8. Crear un DataFrame con la información general de las columnas
        info_df = pd.DataFrame({
            'Column': combined_df.columns,
            'Non-Null Count': combined_df.notnull().sum().values,
            'Dtype': combined_df.dtypes.values
        })

        # 9. Crear un DataFrame con los cálculos (outliers, skewness y kurtosis)
        summary_df = pd.DataFrame({
            'Outliers': outliers_count,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        })

        # 10. Unir la información general con los cálculos
        final_summary_df = pd.concat([info_df.set_index('Column'), summary_df], axis=1).reset_index()

        # 11. Llenar NaN para las columnas no numéricas y target (Y) en los cálculos de skewness, kurtosis y outliers
        final_summary_df.fillna(value={"Outliers": np.nan, "Skewness": np.nan, "Kurtosis": np.nan}, inplace=True)

        display(final_summary_df)

    def datasetVisualization(self, X: DataFrame, Y: DataFrame, title="Dataset Overview"):
        X.head()
        num_features = len(X.columns)  # Número de características
        
        # Crear una figura grande
        fig = plt.figure(figsize=(18, 12))
        
        # Añadir un título general al dashboard
        fig.suptitle(title, fontsize=20, y=1.02)  # `y` ajusta la posición vertical del título

        # Crear una cuadrícula con 3 filas y num_features columnas
        gs = gridspec.GridSpec(3, num_features, height_ratios=[1, 1, 2], width_ratios=[1] * 6)

        # Subgráficos para los histogramas en la parte superior
        for i, col in enumerate(X.columns):
            ax_hist = plt.subplot(gs[0, i])
            sns.histplot(X[col], ax=ax_hist, kde=False, bins=10, color='skyblue', edgecolor='black')

        # Subgráficos para las curvas de densidad (KDE) en la segunda fila    
        for i, col in enumerate(X.columns):
            ax_kde = plt.subplot(gs[1, i])
            sns.kdeplot(X[col], ax=ax_kde, color='skyblue')  # Solo la línea de densidad

        # Subgráfico para los boxplots en la tercera fila, ocupando dos columnas
        ax_boxplots = plt.subplot(gs[2, :2])
        X.plot(kind='box', ax=ax_boxplots, subplots=False, patch_artist=True, grid=True, color='skyblue')
        ax_boxplots.set_title("Features", fontsize=16)

        # Subgráfico para la matriz de correlación, ocupando dos columnas (columnas 2 y 3 de la tercera fila)
        ax_corr = plt.subplot(gs[2, 2:4])
        corr_matrix = X.corr()
        sns.heatmap(corr_matrix, annot=True, ax=ax_corr, cmap='coolwarm', cbar=True)
        ax_corr.set_title("Correlation Matrix", fontsize=16)
        
        # Rotar etiquetas de la matriz de correlación en el eje x
        ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, horizontalalignment='right')

        # Subgráfico para la exploración de las clases de la variable objetivo, ocupando las últimas dos columnas
        ax_labels = plt.subplot(gs[2, 4:])  # Ocupa desde la columna 4 hasta el final
        y_values = Y.value_counts()

        # Gráfico de barras para la variable objetivo
        y_values.plot(kind='bar', ax=ax_labels, color='skyblue')
        ax_labels.set_title("Labels", fontsize=16)
        ax_labels.set_xlabel("Clase")
        ax_labels.set_ylabel("Frecuencia")
        ax_labels.set_xticklabels(ax_labels.get_xticklabels(), rotation=0)

        # Añadir etiquetas encima de las barras
        for p in ax_labels.patches:
            ax_labels.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.01))

        # Ajustar los gráficos
        plt.tight_layout()
        plt.show()

    def trainKNN(self, X, Y, n_neighbors=[5, 10, 20], weights=['uniform', 'distance'], Distance=[1, 2]):
        """
        Función para entrenar un modelo K-Nearest Neighbors (KNN) utilizando búsqueda de hiperparámetros
        con GridSearchCV.

        Parámetros:
        -----------
        X : numpy array o pandas DataFrame
            Matriz de características (features) de los datos de entrenamiento.

        Y : numpy array o pandas Series
            Etiquetas (labels) correspondientes a los datos de entrenamiento.

        n_neighbors : list, opcional
            Lista de enteros que representan el número de vecinos a considerar en el algoritmo KNN.
            Valor por defecto: [5, 10, 20]

        weights : list, opcional
            Lista que especifica el tipo de ponderación que se aplicará a los vecinos.
            Opciones: ['uniform', 'distance']
            Valor por defecto: ['uniform', 'distance']
            - 'uniform': Todos los vecinos tienen la misma importancia.
            - 'distance': Los vecinos más cercanos tienen más peso que los más alejados.

        Distance : list, opcional
            Lista de enteros que definen la distancia que se utilizará en la métrica.
            - p=1: Distancia de Manhattan (L1).
            - p=2: Distancia Euclidiana (L2).
            Valor por defecto: [1, 2]

        Retorno:
        --------
        best_knn_model : KNeighborsClassifier
            El mejor modelo KNN encontrado durante la búsqueda de hiperparámetros utilizando GridSearchCV.
        """

        param_grid = {
            'n_neighbors': n_neighbors,     # Número de vecinos
            'weights': weights,             # Tipo de ponderación de los vecinos
            'p': Distance                   # Distancia: p=1: Manhattan, p=2: Euclidiana
        }

        # Realizar búsqueda de hiperparámetros usando GridSearchCV
        grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
                                param_grid=param_grid,
                                scoring='f1_weighted',
                                cv=3,
                                verbose=0,
                                n_jobs=-1)   

        # Ajustar el modelo con los datos de entrenamiento
        grid_search.fit(X, Y)

        # Obtener los mejores hiperparámetros encontrados
        best_knn_model = grid_search.best_estimator_

        return best_knn_model

    def trainRF(self, X, Y, class_weight, n_estimators=[5, 10, 20, 50], 
                        max_depth=[3, 5, 10], min_samples_split=[3, 5, 10, 20], 
                        min_samples_leaf=[1, 2, 4], bootstrap=[True, False]):
        """
        Función para entrenar un modelo Random Forest utilizando búsqueda de hiperparámetros con GridSearchCV.

        Parámetros:
        -----------
        X : numpy array o pandas DataFrame
            Matriz de características (features) de los datos de entrenamiento.

        Y : numpy array o pandas Series
            Etiquetas (labels) correspondientes a los datos de entrenamiento.

        class_weight : dict
            Diccionario que define los pesos de las clases para compensar el desbalance de datos.

        n_estimators : list, opcional
            Lista de enteros que representan el número de árboles en el bosque.
            Valor por defecto: [5, 10, 20, 50]

        max_depth : list, opcional
            Lista de enteros que especifica la profundidad máxima de los árboles.
            Valor por defecto: [3, 5, 10]

        min_samples_split : list, opcional
            Lista de enteros que define el número mínimo de muestras necesarias para dividir un nodo.
            Valor por defecto: [3, 5, 10, 20]

        min_samples_leaf : list, opcional
            Lista de enteros que especifica el número mínimo de muestras necesarias para estar en una hoja.
            Valor por defecto: [1, 2, 4]

        bootstrap : list, opcional
            Lista que especifica si se utilizará el muestreo con reemplazo (bootstrap).
            Valor por defecto: [True, False]

        Retorno:
        --------
        best_rf_model : RandomForestClassifier
            El mejor modelo Random Forest encontrado durante la búsqueda de hiperparámetros utilizando GridSearchCV.
        """

        # Definir el modelo con los pesos de clase
        random_forest_model = RandomForestClassifier(random_state=42, class_weight=class_weight)

        # Definir el espacio de hiperparámetros para la búsqueda
        param_grid = {
            'n_estimators': n_estimators,               # Número de árboles en el bosque
            'max_depth': max_depth,                     # Profundidad máxima de los árboles
            'min_samples_split': min_samples_split,     # Número mínimo de muestras para dividir un nodo
            'min_samples_leaf': min_samples_leaf,       # Número mínimo de muestras en una hoja
            'bootstrap': bootstrap                      # Usar o no muestreo con reemplazo
        }

        # Realizar búsqueda de hiperparámetros usando GridSearchCV
        grid_search = GridSearchCV(estimator=random_forest_model,
                                param_grid=param_grid,
                                scoring='f1_weighted',  # Puedes cambiar a otra métrica si lo prefieres
                                cv=3,
                                verbose=0,
                                n_jobs=-1)   

        # Ajustar el modelo con los datos de entrenamiento
        grid_search.fit(X, Y)

        # Obtener los mejores hiperparámetros encontrados
        best_rf_model = grid_search.best_estimator_

        return best_rf_model

    def trainAndSaveRFModel(self, X_train, Y_train):
        # Usar 'balanced' para manejar el desbalanceo de clases automáticamente
        class_weight = 'balanced'

        # Entrenar el Modelo Random Forest
        RF_Model = self.trainRF(X_train, Y_train, class_weight)  # Pasar class_weight

        # Guardar el modelo entrenado
        joblib.dump(RF_Model, self.rf_model_filename)

        print(f"Modelo Random Forest guardado exitosamente como {self.rf_model_filename}.")
        return RF_Model

    def trainAndSaveKNNModel(self, X_train, Y_train):
        # Entrenar el Modelo KNN
        KNN_Model = self.trainKNN(X_train, Y_train)

        # Guardar el modelo entrenado
        joblib.dump(KNN_Model, self.knn_model_filename)

        print(f"Modelo KNN guardado exitosamente como {self.knn_model_filename}.")
        return KNN_Model

    def processAndTrainModels(self, X_train, Y_train, X_test, Y_test, label_encoder):
        """
        Realiza el procesamiento de datos, entrena los modelos KNN y Random Forest, 
        evalúa su rendimiento, y guarda el mejor modelo y su desempeño.

        Parámetros:
        -----------
        X_train : array-like
            Los datos de entrenamiento de las características (features).
        
        Y_train : array-like
            Las etiquetas correspondientes a los datos de entrenamiento.
        
        X_test : array-like
            Los datos de prueba de las características (features).
        
        Y_test : array-like
            Las etiquetas correspondientes a los datos de prueba.

        label_encoder : LabelEncoder
            El label encoder utilizado para codificar las etiquetas Y.
        
        Retorno:
        --------
        None
        """
        # Train and save KNN and Random Forest models
        KNN_Model = self.trainAndSaveKNNModel(X_train, Y_train)
        RF_Model = self.trainAndSaveRFModel(X_train, Y_train)

        # Save the label encoder
        self.dumpLabelEncoder(label_encoder)

        # Cargar el modelo entrenado
        KNN_Model = joblib.load(self.knn_model_filename)
        RF_Model = joblib.load(self.rf_model_filename)

        # Cargar el label encoder
        label_encoder = joblib.load(self.label_encoder_filename)

        # Realizar predicciones y evaluar los modelos con X_test
        y_pred_knn = KNN_Model.predict(X_test)
        y_pred_rf = RF_Model.predict(X_test)

        # Evaluar el rendimiento de los modelos
        knn_accuracy = accuracy_score(Y_test, y_pred_knn)
        rf_accuracy = accuracy_score(Y_test, y_pred_rf)

        knn_f1 = f1_score(Y_test, y_pred_knn, average='weighted')
        rf_f1 = f1_score(Y_test, y_pred_rf, average='weighted')

        # Comparar ambos modelos
        better_model = "KNN" if knn_accuracy > rf_accuracy else "Random Forest"
        if knn_accuracy == rf_accuracy:
            better_model = "Both models perform equally"

        # Mostrar el rendimiento de ambos modelos
        print(f"KNN Accuracy: {knn_accuracy}, F1-score: {knn_f1}")
        print(f"Random Forest Accuracy: {rf_accuracy}, F1-score: {rf_f1}")
        print(f"Better Model: {better_model}")

        # Guardar un archivo con la información de qué modelo es mejor y su rendimiento
        with open(self.comparison_filename, "w") as file:
            file.write(f"KNN Accuracy: {knn_accuracy}, F1-score: {knn_f1}\n")
            file.write(f"Random Forest Accuracy: {rf_accuracy}, F1-score: {rf_f1}\n")
            file.write(f"Better Model: {better_model}\n")

        print(f"Información de comparación guardada como {self.comparison_filename}.")
        
    def loadModel(self, version_label = "", model_type=""):
        """
        Cargar el modelo entrenado basado en el version_label y el tipo de modelo especificado.
        
        Parámetros:
        -----------
        version_label : str
            El string que representa la versión de los modelos (ejemplo: "v1.2").
            Si no se pasa, se usara por defecto el pasado por el constructor
        
        model_type : str, opcional
            El tipo de modelo a cargar: "KNN" para K-Nearest Neighbors, "RF" para Random Forest, 
            o si está vacío, carga el mejor modelo basado en el archivo de comparación.

        Retorno:
        --------
        model : sklearn model
            El modelo entrenado (KNN o Random Forest).
            
        label_encoder : LabelEncoder
            El label encoder asociado al modelo.
        """

        if version_label:
            knnModelFilename = f"{self.BASE_DUMP_DIRECTORY}knn_model_{version_label}.pkl"
            rfModelFilename = f"{self.BASE_DUMP_DIRECTORY}rf_model_{version_label}.pkl"
            labelEncoderFilename = f"{self.BASE_DUMP_DIRECTORY}label_encoder_{version_label}.pkl"
            comparisonFilename = f"{self.BASE_DUMP_DIRECTORY}model_comparison_{version_label}.txt"
        else:
            knnModelFilename = self.knn_model_filename
            rfModelFilename = self.rf_model_filename
            labelEncoderFilename = self.label_encoder_filename
            comparisonFilename = self.comparison_filename
            
        
        # Cargar el LabelEncoder
        label_encoder = joblib.load(labelEncoderFilename)
        
        # Si model_type está vacío, cargar el modelo mejor basado en el archivo de comparación
        if model_type == "":
            try:
                with open(comparisonFilename, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        if "Better Model" in line:
                            better_model = line.split(":")[-1].strip()
                            model_type = better_model
                            print(f"El mejor modelo es: {better_model}")
                            break
            except FileNotFoundError:
                raise ValueError(f"No se encontró el archivo de comparación para la versión: {version_label}")
        
        # Cargar el modelo basado en el tipo especificado (KNN o RF)
        if model_type == "KNN":
            model = joblib.load(knnModelFilename)
            print(f"Modelo KNN cargado desde {knnModelFilename}.")
        elif model_type == "RF" or model_type == "Random Forest":
            model = joblib.load(rfModelFilename)
            print(model)
            print(f"Modelo Random Forest cargado desde {rfModelFilename}.")
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {model_type}. Debe ser 'KNN', 'RF' o 'Random Forest'.")
        
        return model, label_encoder

