from transformers import BertTokenizerFast
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm 
import pickle
import torch 



class SquadDataset(torch.utils.data.Dataset):
  def __init__(self,encodings):
    self.encodings = encodings
  def __getitem__(self,idx):
     return {key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)
def add_token_positions(encodings,answers,tokenizer,train_dataset):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i,train_dataset["answers"][i]['answer_start']))
    end_positions.append(encodings.char_to_token(i,train_dataset["answers"][i]['answer_end']))
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    go_back = 1
    while end_positions[-1] is None:
      end_positions[-1] = encodings.char_to_token(i,train_dataset["answers"][i]['answer_end']-go_back)
      go_back += 1
  encodings.update({'start_positions':start_positions,'end_positions':end_positions})  
  return encodings
def get_dataset(tr_enc,tst_enc):
  train_dataset_for_model = SquadDataset(tr_enc)
  test_dataset_for_model = SquadDataset(tst_enc)
  return train_dataset_for_model,test_dataset_for_model
def get_encodings(tokenizer_name,number_of_rows_data,train_dataset,test_dataset):
  tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
  train_encodings = tokenizer(train_dataset["contexts"][0:number_of_rows_data],train_dataset["questions"][0:number_of_rows_data],truncation=True,padding=True)
  test_encodings = tokenizer(test_dataset["contexts"][0:number_of_rows_data],test_dataset["questions"][0:number_of_rows_data],truncation=True,padding=True)
  train_encodings = add_token_positions(train_encodings,train_dataset["answers"][0:number_of_rows_data],tokenizer,train_dataset)
  test_encodings = add_token_positions(test_encodings,test_dataset["answers"][0:number_of_rows_data],tokenizer,train_dataset)
  return train_encodings,test_encodings
def fine_tune_qna_bert(model_name,tokenizer_name,epochs,train_dataset,test_dataset,number_of_rows_data):
  train_encodings,test_encodings = get_encodings(tokenizer_name,number_of_rows_data,train_dataset,test_dataset)
  train_dataset_for_model,test_dataset_for_model = get_dataset(train_encodings,test_encodings)
  tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
  model = BertForQuestionAnswering.from_pretrained(model_name)
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)
  model.train()
  # model.eval()
  optim = AdamW(model.parameters(),lr=05e-5)
  train_loader = DataLoader(train_dataset_for_model,batch_size=8,shuffle = True)
  for epoch in range(epochs):
    loop = tqdm(train_loader)
    for batch in loop:
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_positions = batch['start_positions'].to(device)
      end_positions = batch['end_positions'].to(device)
      
      outputs = model(input_ids,attention_mask=attention_mask,start_positions=start_positions,end_positions=end_positions)
      loss = outputs[0]
      loss.backward()
      optim.step()
      loop.set_description(f'Epoch: {epoch}')
      loop.set_postfix(loss=loss.item())
  # model_path = f"model/{model_name}/{epochs}/{number_of_rows_data}"
  # model.save_pretrained(model_path)
  # tokenizer.save_pretrained(model_path)
  return model,test_dataset_for_model,device,tokenizer

def pickle_save(model,model_path,model_name):
  pickle.dump(model,open(model_path + f"{model_name}.pkl",'wb'))