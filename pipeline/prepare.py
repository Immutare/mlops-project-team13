import os
import random
import sys
import pandas as pd
import yaml



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
    output_train = os.path.join("data", "prepared", "train.csv")
    output_test = os.path.join("data", "prepared", "test.csv")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)


    dataframe = pd.read_csv('dataset/MaternalHealthRiskDataSet.csv')
    trainDataframe = dataframe.sample(frac=trainSplit,random_state=200)
    testDataframe = dataframe.drop(trainDataframe.index).sample(frac=1.0)

    trainDataframe.to_csv(output_train, index=False)
    testDataframe.to_csv(output_test, index=False)

if __name__ == "__main__":
    main()
