import os
import pickle

import spacy
from transformers import AutoTokenizer, AutoModel
def tokenize(path):
    nlp = spacy.load("en_core_web_sm")
    for file in os.listdir(path):
        text = ""
        with open(path+file, "r") as f:
            text = f.read()
        doc = nlp(text)
        i = 0
        spacy_tok_sents = []
        sent = []
        first = True
        for token in doc:
            if first:
                if token.is_space == False:
                    sent.append(token.text)
                first = False
            elif token.is_sent_start:
                spacy_tok_sents.append(sent)
                sent = []
                if token.is_space == False:
                    sent.append(token.text)
            else:
                if token.is_space == False:
                    sent.append(token.text)
        if len(sent)!=0:
            spacy_tok_sents.append(sent)
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_tokens_w_offsets = tokenizer(spacy_tok_sents,padding=True, truncation=True, max_length=512, return_tensors="pt", is_split_into_words=True, return_offsets_mapping=True)
        bert_tokens = tokenizer(spacy_tok_sents, padding=True, truncation=True, max_length=512,
                                          return_tensors="pt", is_split_into_words=True)
        bert_outputs = model(**bert_tokens)
        print(bert_outputs)
        exit()

        doc.user_data = {}
        doc.user_data["subwords"] = []
        i=1
        j=0
        # for more than one doc, for doc in docs?
        subwords = []
        first = True
        finished = True
        prev = ""
        tcounter = 0
        print(bert_tokens.input_ids.size())

        for token in doc:
            bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
            if token.text.lower().startswith("1826"):
                print("howdy")
            while (j<bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102):
                i = 1
                j += 1
                #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
            if j==bert_tokens.input_ids.size()[0]:
                doc.user_data["subwords"].append(subwords)
                break
            if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) and finished==True:
                if token.text.lower() != tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).lower():
                    finished=False
                else:
                    finished = True
                if first == False:
                    doc.user_data["subwords"].append(subwords)
                    subwords=[]
                subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
            elif tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##") or finished==False:
                subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                i+=1
                bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                temp = ""+subwords[0].lower() +subwords[1].lower()
                if temp.replace("##","").lower()==prev.lower():
                    finished = True
                while(token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))==False and (tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##") or finished==False)):
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                    i+=1
                    bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                    temp+=subwords[len(subwords)-1].lower()
                    print(temp.lower().replace("##",""))
                    if prev.lower() == temp or prev.lower()==temp.lower().replace("##",""):
                        finished=True
                finished=True
                while (j<bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102):
                    i = 1
                    j += 1
                    bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))):
                    if token.text.lower() != tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).lower():
                        finished = False
                    else:
                        finished = True
                    if first == False:
                        doc.user_data["subwords"].append(subwords)
                        subwords = []
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                else:
                    doc.user_data["subwords"].append(subwords)
                    subwords = []

                    i -= 1
                    bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
            else:
                doc.user_data["subwords"].append(subwords)
                subwords = []
                finished=True
                i-=1
                bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
            i+=1
            bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
            first = False
            prev = token.text
        diff = len(doc) - len(doc.user_data["subwords"])
        for i in range(len(doc)-diff, len(doc)):
            doc.user_data["subwords"].append([])
        # print(len(doc))
        # print("here they are:"+doc[4611].text+doc[4612].text+doc[4613].text)
        # print(len(doc.user_data["subwords"]))
        # exit()
        i=0
        for token in doc:
            print("Word: "+token.text+"\tSubwords: "+str(doc.user_data["subwords"][i]))
            i+=1