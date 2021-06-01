import os
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
        print(spacy_tok_sents[0])
        bert_tokens = tokenizer(spacy_tok_sents[0],padding=True, truncation=True, max_length=512, return_tensors="pt", is_split_into_words=True, return_offsets_mapping=True)
        print(bert_tokens.offset_mapping)

