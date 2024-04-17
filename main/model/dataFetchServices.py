import pymysql 
import pandas as pd
from os import path
import json
from sklearn.model_selection import train_test_split
from model import mapping as m
import os

def fetch_data(): #1
    return pd.read_csv("../../file.csv")
def toJson(csvFilePath,metadata):
    if metadata["task_name"] == "qna":
        Df = pd.read_csv(csvFilePath, na_filter=False, dtype=str)
        Df["context"] = Df.title.str.cat(Df[metadata["cotext_name"]], sep=" ")
        # Df.drop(["text", "title"], axis=1)
        Df = Df[["context",  metadata["answer"]]]
        # print(Df)
        Df["context"] = Df["context"].apply(lambda x:str(x).lower())
        Df[metadata["answer"]] = Df["Companies"].apply(lambda x:str(x).lower())
        # parent_dict = {"root":}
        data = []
        questions = metadata["questions"]
        for _ , row in Df.iterrows():
            newDict = {"qas": [], "context": str(row["context"])}
            id = 1
            for question in questions:
                tempDict = {
                    "id": str(id),
                    "is_impossible": False,
                    "question": question
                }
                
                tempDict["answers"] = [
                    {
                        "text": row["Companies"], #1
                        "answer_start": str(row["context"]).lower().find(str(row["Companies"]).lower())
                    }
                ]
            
                if str(row["context"]).lower().find(str(row["Companies"])) != -1:
                    newDict["qas"].append(tempDict)
                    id += 1
            data.append(newDict)
        para_dict = {"title":"pre ipo data","paragraphs":data}
        data_dict = {"data":[para_dict]}
        final_data_for_model = {"root":data_dict}

        fileName = "squad/" + path.basename(csvFilePath).split(sep=".")[0] + "_data1.json"
        with open(fileName, "w") as td:
            json.dump(data_dict, td, indent=4, sort_keys=True)

def create_datasets():
    df = fetch_data()
    train,test = train_test_split(df,test_size=0.2)
    train.to_csv("training.csv")
    test.to_csv("test.csv")
    print(df.head())
    toJson("training.csv",m.tasks)
    toJson("test.csv",m.tasks)

if __name__ == "__main__":
    print(setup_connection())