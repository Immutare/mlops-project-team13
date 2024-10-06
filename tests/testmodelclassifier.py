import pandas as pd
from sklearn.model_selection import train_test_split
from model.classifiermodel import ClassifierModel

class TestModelClassifier:
    ACCURACY_THRESHOLD = 0.75
    PRECISION_THRESHOLD = 0.75

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

    def __init__(self, randomSeed):
        self.RANDOM_STATE_SEED = randomSeed
        self._model = None
        self._dataframe = None
        self._X = None
        self._y = None

    def getDataframe(self): 
        # Loads the dataset
        self._dataframe = pd.read_csv('../dataset/Maternal Health Risk Data Set.csv')

        self._y = self._dataframe[['RiskLevel']].copy()
        self._X = self._dataframe.drop(columns=['RiskLevel'])
    

    def test_datapreprocessing_is_data_clean(self):
        if (self._model is None):
            self._model = ClassifierModel()
        if (self._dataframe is None):
            self.getDataframe()

        errors = []
        if not self._model.isnull().values.any() == False:
            errors.append("there are null values on the data")
        # TODO: Add outliers to the list of testing
        # if not condition_2:
        #     errors.append("an other error message")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

        self._model.cleanAndPreprocessData(self._X, self._y)

        assert self._model.isnull().values.any() == False



    
    def test_classification_metrics(self):
        if (self._model is None):
            self._model = ClassifierModel()
        # Split the model
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.1, random_state=self.RANDOM_STATE_SEED)

        # Train the model
        self._model.searchBestEstimator(X_train, y_train)

        # Make predictions using your classification algorithm
        predictions = self._model.predict(X_test)

        # Calculate accuracy
        correct_predictions = (predictions == y_test).sum()
        total_predictions = len(X_test)
        accuracy = correct_predictions / total_predictions

        # TODO: Calculate precision
        # true_positives = ((predictions == 1) & (test_data['actual_labels'] == 1)).sum()
        # false_positives = ((predictions == 1) & (test_data['actual_labels'] == 0)).sum()
        # precision = true_positives / (true_positives + false_positives)

        # Assert that accuracy and precision meet your criteria
        assert accuracy >= self.ACCURACY_THRESHOLD
        assert precision >= self.PRECISION_THRESHOLD

    def test_two(self):
        assert self.value == 1