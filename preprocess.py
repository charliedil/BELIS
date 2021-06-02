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

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_tokens = tokenizer(spacy_tok_sents,padding=True, truncation=True, max_length=512, return_tensors="pt", is_split_into_words=True, return_offsets_mapping=True)
        print(bert_tokens)

        doc.user_data = {}
        doc.user_data["subwords"] = []
        i=1
        j=0
        # for more than one doc, for doc in docs?
        subwords = []
        first = True
        for token in doc:
            if int(bert_tokens.input_ids[j][i]) == 102:
                i = 1
                j += 1
            if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))):
                if first == False:
                    doc.user_data["subwords"].append(subwords)
                    subwords=[]
                subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
            elif tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##"):
                subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                i+=1
                while(token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))==False and tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##")):
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                    i+=1
                if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))):
                    if first == False:
                        doc.user_data["subwords"].append(subwords)
                        subwords = []
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                else:
                    doc.user_data["subwords"].append(subwords)
                    subwords = []

                    i -= 1
            else:
                doc.user_data["subwords"].append(subwords)
                subwords = []

                i-=1
            i+=1
            first = False
        i=0
        for token in doc:
            print("Word: "+token.text+"\tSubwords: "+str(doc.user_data["subwords"][i]))
            i+=1