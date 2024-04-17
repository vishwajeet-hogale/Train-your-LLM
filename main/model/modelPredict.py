import torch
questions = ["which company is going ipo?","which compay is going to be listed?"]
# torch.cuda.empty_cache()
# myModel.to(device)
def question_answer(question, text,tokenizer,device,myModel):
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    # segment_ids.to(device)
    token_type_ids = torch.tensor([segment_ids]).to(device)
    #model output using input_ids and segment_ids
    i_ids = torch.tensor([input_ids]).to(device)
    # print(token_type_ids)
    # print(i_ids)
    output = myModel(i_ids, token_type_ids=token_type_ids)
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    else:
      answer= "[CLS]"          
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    return answer.capitalize()
def QnA(input_dir, output_dir,tokenizer,device,myModel,file_name='todays_report.csv'):
        import os
        print(os.getcwd())
        # %%
        import pandas as pd
        # import spacy
        from datetime import datetime ,date
        cur_date = str(date.today())
        input_file_fullpath = os.path.join(input_dir,file_name)
        # logging.info("reading input artifact " + input_file_fullpath)
        data = pd.read_csv(input_file_fullpath, encoding='utf8')
        # logging.info("completed reading input artifact " + input_file_fullpath)
        # data.drop('companies', inplace=True, axis=1)
        data = data[['text','publish_date','scraped_date','title','link','Companies']]
        text = data['text']
        title = data['title']
    
        company_names = []
        for val,i in data.iterrows():
            t = str(i["title"]) + " " + str(i["text"])
            # print(t)
            try:
              c = question_answer(questions[0],t,tokenizer,device,myModel)
              print(c)
              company_names.append(c)
            except:
              t = t[0:512]
              c = question_answer(questions[0],t,tokenizer,device,myModel)
              company_names.append(c)
        # data["Companies"] = company_names
        dic = {'Companies':company_names}
        #["abc","def"]
        df = pd.DataFrame(dic)
        #data=["text","title"]
        dff = [data,df]
        
        df_final = pd.concat(dff,axis=1)
        # df_final.to_csv('EDI_PREIPO_report.csv',index=False)
        
        edi_preipo_report_fname = os.path.join( output_dir, 'EDI_PREIPO_report.csv' )
        # logging.info("writing output artifact " + edi_preipo_report_fname + " to " + output_dir)
        df_final.to_csv(edi_preipo_report_fname,index=False)
        # logging.info("completed writing output artifact " + edi_preipo_report_fname + " to " + output_dir)
        return df_final