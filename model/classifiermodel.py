import pandas as pd
from pandas import DataFrame
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from .informalmodelinterface import InformalModelInterface

class ClassifierModel (InformalModelInterface):    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, randomSeed):
        self.RANDOM_STATE_SEED = randomSeed
        self._model = None

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

    def searchBestEstimator(self, X_train, y_train):
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
    
    def predict(self, X):
        if self._model is not None:
            return self._model.predict(X)
        raise Exception("Couldn't find a model")

