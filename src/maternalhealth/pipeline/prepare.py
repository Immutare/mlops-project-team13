import os
import random
import sys
import pandas as pd
from pandas import DataFrame
import yaml

def disproportionateSampling(dataframe: DataFrame, target: str = ''):
    dataframe.groupby('Grade', group_keys=False).apply(lambda x: x.sample(2))




def split_test_train(dataframe: DataFrame, trainSplit = 0.8, seed = 400):
    # disproportionateDF = disproportionateSampling(dataframe, )
    trainDataframe = dataframe.sample(frac=trainSplit,random_state=seed)
    testDataframe = dataframe.drop(trainDataframe.index).sample(frac=1.0)

    return trainDataframe, testDataframe



def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1) 

    # Test data set split ratio
    trainSplit = params["train_split"]
    trainSplit = 1 - trainSplit
    random.seed(params["seed"])

    input = sys.argv[1]
    output_train_path = os.path.join("data", "prepared", "train.csv")
    output_test_path = os.path.join("data", "prepared", "test.csv")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    dataframe = pd.read_csv(input)

    trainDataframe, testDataframe = split_test_train(dataframe)

    trainDataframe.to_csv(output_train_path, index=False)
    testDataframe.to_csv(output_test_path, index=False)

if __name__ == "__main__":
    main()
