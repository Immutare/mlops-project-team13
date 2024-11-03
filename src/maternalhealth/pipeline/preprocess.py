import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import LabelEncoder

from maternalhealth.pipeline.model.classifiermodel import ClassifierModel


def getDataSet(inputFolder, fileName):
    """
    This should be the last dataset we got from the previous step
    """
    filePath = os.path.join(inputFolder, fileName)
    dataframe = pd.read_csv(filePath, encoding="utf-8")
    sys.stderr.write(
        f"The input data frame {fileName} size is {dataframe.shape}\n"
    )

    return dataframe


def transformDF(
    dataframe: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, LabelEncoder]:
    classifierModel = ClassifierModel(
        versionLabel=kwargs.get("version_label") or "0.0",
        randomSeed=kwargs.get("seed"),
        asTesting=kwargs.get("testing") == "Y",
    )

    # Split the dataset from numeric
    numericColumns = dataframe.drop(
        dataframe.select_dtypes(exclude=[np.number]).columns, axis=1
    )

    numericTransformed = classifierModel.outlierHandler(numericColumns)
    numericTransformed = classifierModel.applyTransformations(
        numericTransformed
    )
    numericTransformed = classifierModel.normalizeData(numericTransformed)

    # Transform target
    nonNumeric = dataframe.drop(
        dataframe.select_dtypes(include=[np.number]).columns, axis=1
    )
    targetColumn = nonNumeric[["RiskLevel"]].copy()
    # bypass = nonNumeric.drop("RiskLevel", axis=1)

    targetTransformed, labelEncoder = classifierModel.labelEncoding(
        targetColumn
    )

    transformedDataframe = pd.concat(
        [
            numericTransformed,
            pd.Series(targetTransformed, dtype=int).rename("RiskLevel"),
        ],
        axis=1,
    )

    return transformedDataframe, labelEncoder


def saveLabelEncoder(labelEncoder, outputFolder, fileName):
    outputPath = os.path.join(outputFolder, fileName)

    joblib.dump(labelEncoder, outputPath)
    sys.stderr.write(f"LabelEncoder [{fileName}] saved on {outputPath}\n")


def saveTransformedDataFrame(
    transformedDF: pd.DataFrame, outputFolder, fileName
):
    outputPath = os.path.join(outputFolder, fileName)

    transformedDF.to_csv(outputPath, index=False)
    sys.stderr.write(f"Dataframe [{fileName}] saved on {outputPath}\n")


def main():
    params = yaml.safe_load(open("params.yaml"))
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython featurization.py data-dir-path features-dir-path\n"
        )
        sys.exit(1)

    in_path = sys.argv[1]  # previous step folder
    out_path = sys.argv[2]  # output folder

    os.makedirs(out_path, exist_ok=True)

    trainInputDF = getDataSet(in_path, "train.csv")
    testInputDF = getDataSet(in_path, "test.csv")

    transformedTrainDF, trainLabelEncoder = transformDF(
        dataframe=trainInputDF, **params
    )
    transformedTestDF, testLabelEncoder = transformDF(
        dataframe=testInputDF, **params
    )

    saveTransformedDataFrame(transformedTrainDF, out_path, "train.csv")
    saveLabelEncoder(trainLabelEncoder, out_path, "train_label_encoder.pkl")

    saveTransformedDataFrame(transformedTestDF, out_path, "test.csv")
    saveLabelEncoder(testLabelEncoder, out_path, "test_label_encoder.pkl")

    sys.stderr.write("---Pipeline finished---\n")


if __name__ == "__main__":
    main()
