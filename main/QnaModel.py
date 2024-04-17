import os
import pymysql 
import pandas as pd
import torch
from os import path
import json
from sklearn.model_selection import train_test_split
from model.dataFetchServices import *
from model.modelServices import *
from model.modelPredict import *
from transformers import BertTokenizerFast
class QnAModel():
    
    def __init__(self):
        try:
            os.mkdir('squad')
        except:
            print("Folder already exists")
    def fetch_data_from_db(self):
        create_datasets()
    def data_prep_for_model(self):
        def read_squad(path):
            with open(path,'rb') as f:
                squad_dict = json.load(f)
            contexts = []
            questions = []
            answers = []
            for group in squad_dict['data']:
                for passage in group['paragraphs']:
                    context = passage["context"]
                    for qa in passage['qas']:
                        question = qa["question"]
                        for answer in qa["answers"]:
                            contexts.append(context)
                            questions.append(question)
                            answers.append(answer)
            return {"contexts":contexts,"questions":questions,"answers":answers}
        train_dataset = read_squad("./squad/training_data1.json")
        test_dataset = read_squad("./squad/test_data1.json")
        def add_end_index(answers,contexts):
            for answer,context in zip(answers,contexts):
                gold_text = answer['text']
                start_idx = answer['answer_start']
                end_idx = start_idx+len(gold_text)
                if context[start_idx:end_idx] == gold_text:
                    answer['answer_end'] = end_idx
                else:
                    for n in [1,2]:
                        if(context[start_idx-n:end_idx-n] == gold_text):
                            answer['answer_end'] = end_idx-n
                            answer['answer_start'] = start_idx-n
        add_end_index(train_dataset["answers"],train_dataset["contexts"])
        add_end_index(test_dataset["answers"],test_dataset["contexts"])
        return train_dataset,test_dataset
    def fine_tune_train(self,train_dataset,test_dataset,model_name='bert-base-uncased',tokenizer_name='bert-base-uncased',epochs=10,number_of_rows_data = 2000):
        return fine_tune_qna_bert(model_name,model_name,epochs=epochs,train_dataset=train_dataset,test_dataset=test_dataset,number_of_rows_data = number_of_rows_data)
    def predict_on_dataframe(self,input_dir, output_dir,tokenizer,device,myModel,file_name):
        return QnA(input_dir,output_dir,tokenizer,device,myModel,file_name)
    def save(self,model,model_path,model_name):
        return pickle_save(model,model_path,model_name)
    def load(self,model_path,device):
        with open(model_path, "rb") as newFile:
            myModel = torch.load(newFile)
            myModel.to(device)
        return myModel
    


if __name__ == "__main__":
    # bert = QnAModel()
    # bert.fetch_data_from_db()
    # train_dataset,test_dataset = bert.data_prep_for_model()
    # model,test_dataset_for_model,device,tokenizer = bert.fine_tune_train(train_dataset=train_dataset,test_dataset=test_dataset,model_name='bert-base-uncased',tokenizer_name='bert-base-uncased',epochs=10,number_of_rows_data = 2000)
    # bert.predict_on_dataframe("","",tokenizer,device,model)
    

    ## Test 
    bert_uncased = QnAModel()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    df = bert_uncased.load_predict_dataframe("","",tokenizer,device,"bert-base-uncased.pkl")
    