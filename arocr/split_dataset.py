import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(datset, file):
    print ("loading the data")
    df = pd.read_csv(file, sep="\t")
    df_train, df_dev_test =  train_test_split(df, test_size= 0.2, random_state=41)
    df_test, df_dev =  train_test_split(df_dev_test, test_size= 0.5, random_state=41)
    print ("saving the split")
    df_train.to_csv("./data/{}/".format(dataset_name)+"/train.tsv", sep="\t", index=None)
    df_dev.to_csv("./data/{}/".format(dataset_name)+"/valid.tsv", sep="\t", index=None)
    df_test.to_csv("./data/{}/".format(dataset_name)+"/test.tsv", sep="\t", index=None)



print ("start")
dataset_name="OnlineKhatt"
split_dataset(dataset_name, "./data/{}/{}.tsv".format(dataset_name,dataset_name))