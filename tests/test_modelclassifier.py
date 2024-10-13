from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from model.classifiermodel import ClassifierModel
from pandas import DataFrame
import numpy as np

class TestModelClassifier:
    ACCURACY_THRESHOLD = 0.75
    PRECISION_THRESHOLD = 0.75
    _tag = "tester_{date}".format(date=datetime.today().strftime("%B%d_%Y__%H%M%S"))
    _model = None
    _dataframe = None
    _X = None
    _y = None

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value

    @property
    def dataframe(self):
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = value

    def getDataframe(self, shuffle = False): 
        # Loads the dataset
        self._dataframe = pd.read_csv('dataset/Maternal Health Risk Data Set.csv')

        if (shuffle):
            self._dataframe.sample(n=len(self._dataframe))

        self._y = self._dataframe[['RiskLevel']].copy()
        self._X = self._dataframe.drop(columns=['RiskLevel'])

    def getOutliersFromDataframe(self, dataframe: DataFrame, threshold = 1.5):
        if (self._model is None):
            self._model = ClassifierModel(self._tag, asTesting = True)
        
        threshold = self._model.IQ_LOWER_RANGE
        numericColumns = dataframe.select_dtypes(include=[np.number]).columns

        outliersDict = dict()

        for column in numericColumns:
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)
            IQR = Q3 - Q1

            # identify outliers
            outliers = dataframe.where((dataframe[column] < (Q1 - threshold * IQR)) | (dataframe[column] > (Q3 + threshold * IQR)))
            outliersCount = len(dataframe) - outliers[column].isna().sum()

            outliersDict[column] = int(outliersCount)
            print(column, outliersCount)
        return outliersDict
    
    def test_handleOutliers(self):
        if (self._model is None):
            self._model = ClassifierModel(self._tag, asTesting = True)
        if (self._dataframe is None):
            self.getDataframe()
    
        dataframe = self._dataframe.copy().select_dtypes(include=[np.number])

        hasOutliers = True
        outliersBeforePreporcessing = self.getOutliersFromDataframe(dataframe)
        dfWithNoOutliers = self._model.outlierHandler(dataframe)
        outliersAfterPreporcessing = self.getOutliersFromDataframe(dfWithNoOutliers)

        for column in dataframe.select_dtypes(include=[np.number]).columns:
            print(f"outliersBefore - {column}: ", outliersBeforePreporcessing[column])
            print(f"outliersAfter - {column}: ", outliersAfterPreporcessing[column])
            if outliersAfterPreporcessing[column] > 0:
                hasOutliers = False
                break

        # assert no error message has been registered, else print messages
        assert hasOutliers

    '''
    def test_datapreprocessing_is_data_clean(self, threshold = 1.5):
        if (self._model is None):
            self._model = ClassifierModel(self._tag, asTesting = True)
        if (self._dataframe is None):
            self.getDataframe()
        dataframe = self._dataframe.copy()

        errors = []
        if not dataframe.isnull().values.any() == False:
            errors.append("there are null values on the data")
        
        hasOutliers = True
        outliersBeforePreporcessing = self.getOutliersFromDataframe(dataframe)
        dfWithNoOutliers = self._model.outlierHandler(dataframe)
        outliersAfterPreporcessing = self.getOutliersFromDataframe(dfWithNoOutliers)

        for column in dataframe.select_dtypes(include=[np.number]).columns:
            if outliersBeforePreporcessing[column] == outliersAfterPreporcessing[column]:
                hasOutliers = False
                break

        # TODO: Add outliers to the list of testing
        if hasOutliers:
            errors.append("there are outliers values on the data")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

        self._model.cleanAndPreprocessData(self._X, self._y)

        assert self._model.isnull().values.any() == False
    '''
    def measureClassifierAccuracy (self, model, X,  y, labelEncoder, acceptanceTreshold = 0.75) -> bool:
        # Evaluar el modelo con el conjunto de prueba
        y_pred = model.predict(X)

        # Obtener el reporte de clasificación
        report = classification_report(y, y_pred, target_names=labelEncoder.classes_, output_dict=True)

        return report['accuracy'] >= acceptanceTreshold

    def test_TrainAndSaveRFModel(self):
        if (self._dataframe is None):
            self.getDataframe()
        if (self._model is None):
            self._model = ClassifierModel(self._tag, asTesting = True)


        Xt = self._model.normalizeData(self._model.applyTransformations(self._model.outlierHandler(self._X)))
        
        yt, _ = self._model.labelEncoding(self._y, True)
        
        X_train, X_test, Y_train, Y_test = train_test_split(Xt, yt, test_size=0.2, random_state=self._model.RANDOM_STATE_SEED)
        
        self._model.trainAndSaveRFModel(X_train, Y_train)

        isTestSuccessfull = False
        try:
            model, labelEncoder = self._model.loadModel(model_type='RF')
            isTestSuccessfull = self.measureClassifierAccuracy(model, X_test, Y_test, labelEncoder)
        except:
            print("Something went wrong loading the model")
        
        assert isTestSuccessfull, "model accuracy didn't reach above 80%"

    def test_TrainAndSaveKNNModel(self):
        if (self._dataframe is None):
            self.getDataframe()
        if (self._model is None):
            self._model = ClassifierModel(self._tag, asTesting = True)
        

        Xt = self._model.normalizeData(self._model.applyTransformations(self._model.outlierHandler(self._X)))
        yt, _ = self._model.labelEncoding(self._y, True)
        
        X_train, X_test, Y_train, Y_test = train_test_split(Xt, yt, test_size=0.2, random_state=self._model.RANDOM_STATE_SEED)
        
        _ = self._model.trainAndSaveKNNModel(X_train, Y_train)

        isTestSuccessfull = False
        try:
            model, labelEncoder = self._model.loadModel(model_type='KNN')
            isTestSuccessfull = self.measureClassifierAccuracy(model, X_test, Y_test, labelEncoder)
        except:
            print("Something went wrong loading the model")
        
        assert isTestSuccessfull, "model accuracy didn't reach above 80%"