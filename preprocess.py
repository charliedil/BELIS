import os
import pickle

import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from transformers import AutoTokenizer, AutoModel


def tokenize(path):
    nlp = spacy.load("en_core_web_sm")
    doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
    for file in os.listdir(path):
        if file.endswith(".txt"):
            text = ""
            with open(path + file, "r") as f:
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
            if len(sent) != 0:
                spacy_tok_sents.append(sent)
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            bert_offsets = tokenizer(spacy_tok_sents, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt", is_split_into_words=True,
                                     return_offsets_mapping=True).offset_mapping

            bert_tokens = tokenizer(spacy_tok_sents, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt", is_split_into_words=True)
            bert_outputs = model(**bert_tokens).last_hidden_state.detach()

            doc.user_data = {}
            doc.user_data["filename"] = file
            doc.user_data["subwords"] = []
            doc.user_data["subword_embeddings"] = []
            doc.user_data["spans"] = []
            doc.user_data["subword_spans"] = []
            i = 1
            j = 0
            # for more than one doc, for doc in docs?
            subwords = []
            subword_embeddings = []
            subword_spans = []
            first = True
            finished = True
            prev = ""
            tcounter = 0

            for token in doc:
                doc.user_data["spans"].append((token.idx, len(token.text) + token.idx))
                while (j < bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102):
                    i = 1
                    j += 1
                    # bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                if j == bert_tokens.input_ids.size()[0]:
                    doc.user_data["subwords"].append(subwords)
                    doc.user_data["subword_embeddings"].append(subword_embeddings)
                    doc.user_data["subword_spans"].append(subword_spans)
                    break
                if token.text.lower().startswith(
                        tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) and finished == True:
                    if token.text.lower() != tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).lower():
                        finished = False
                    else:
                        finished = True
                    if first == False:
                        doc.user_data["subwords"].append(subwords)
                        subwords = []
                        doc.user_data["subword_embeddings"].append(subword_embeddings)
                        subword_embeddings = []
                        doc.user_data["subword_spans"].append(subword_spans)
                        subword_spans = []
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                    subword_embeddings.append(bert_outputs[j][i].detach().numpy())
                    subword_spans.append((int(bert_offsets[j][i][0]) + token.idx,
                                          int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + token.idx))
                elif tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith(
                        "##") or finished == False:
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                    subword_embeddings.append(bert_outputs[j][i].detach().numpy())
                    if tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##"):
                        subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx - 3))
                    else:
                        subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx))
                    i += 1
                    if i < bert_tokens.input_ids[j].size()[0]:
                        print(bert_tokens.input_ids[j].size()[0])
                        bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                        temp = "" + subwords[0].lower() + subwords[1].lower()
                        if temp.replace("##", "").lower() == prev.lower():
                            finished = True
                        while (token.text.lower().startswith(
                                tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) == False and (
                                    tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith(
                                           "##") or finished == False)):
                            subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                            subword_embeddings.append(bert_outputs[j][i].detach().numpy())
                            if tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##"):
                                subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                                  int(bert_offsets[j][i][1]) + int(
                                                      bert_offsets[j][i][0]) + prev_idx - 3))
                            else:
                                subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                                  int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx))
                            i += 1
                            #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                            temp += subwords[len(subwords) - 1].lower()
                            if prev.lower() == temp or prev.lower() == temp.lower().replace("##", ""):
                                finished = True
                        finished = True
                        while (j < bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102):
                            i = 1
                            j += 1
                            bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                        if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))):
                            if token.text.lower() != tokenizer.convert_ids_to_tokens(
                                int(bert_tokens.input_ids[j][i])).lower():
                                finished = False
                            else:
                                finished = True
                            if first == False:
                                doc.user_data["subwords"].append(subwords)
                                doc.user_data["subword_embeddings"].append(subword_embeddings)
                                doc.user_data["subword_spans"].append(subword_spans)
                                subwords = []
                                subword_embeddings = []
                                subword_spans = []
                            subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                            subword_embeddings.append(bert_outputs[j][i].detach().numpy())
                            subword_spans.append((int(bert_offsets[j][i][0]) + token.idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + token.idx))
                        else:
                            doc.user_data["subwords"].append(subwords)
                            doc.user_data["subword_embeddings"].append(subword_embeddings)
                            doc.user_data["subword_spans"].append(subword_spans)
                            subwords = []
                            subword_embeddings = []
                            subword_spans = []

                            i -= 1
                            bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                else: #HERE
                    doc.user_data["subwords"].append(subwords)
                    doc.user_data["subword_embeddings"].append(subword_embeddings)
                    doc.user_data["subword_spans"].append(subword_spans)
                    subwords = []
                    subword_embeddings = []
                    subword_spans = []
                    finished = True
                    i -= 1
                    bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                i += 1
                bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                first = False
                prev = token.text
                prev_idx = token.idx
            diff = len(doc) - len(doc.user_data["subwords"])
            for i in range(len(doc) - diff, len(doc)):
                doc.user_data["subwords"].append([])
                doc.user_data["subword_embeddings"].append([])
                doc.user_data["subword_spans"].append([])
            doc.user_data["ents"] = []
            for i in range(len(doc.user_data["subwords"])):
                labels = []
                for j in range(len(doc.user_data["subwords"][i])):
                    if j == 0:
                        labels.append("B-Other")
                    else:
                        labels.append("I-Other")
                doc.user_data["ents"].append(labels)
            with open(path + file.split(".")[0] + ".ann", "r") as f:
                lines = f.read().split("\n")
                for l in lines:
                    if l.startswith("T"):
                        entity = l.split("\t")[1].split(" ")[0]
                        start_span = 0
                        end_span=0
                        if ";" in l.split("\t")[1]:
                            start_span = int(l.split("\t")[1].split(" ")[1])
                            end_span = int(l.split("\t")[1].split(" ")[3])
                        else:
                            start_span = int(l.split("\t")[1].split(" ")[1])
                            end_span = int(l.split("\t")[1].split(" ")[2])
                        for i in range(len(doc.user_data["spans"])):
                            spacy_span = doc.user_data["spans"][i]
                            if spacy_span[0] <= start_span and spacy_span[1] >= end_span:
                                labels = []
                                for j in range(len(doc.user_data["subwords"][i])):
                                    if j == 0:
                                        labels.append("B-" + entity)
                                    else:
                                        labels.append("I-" + entity)
                                doc.user_data["ents"][i] = labels
                                break

            doc_bin.add(doc)
    doc_bin.to_disk("BELIS/datasets/n2c2_train_labeled.spacy")
    nlp.vocab.to_disk("BELIS/datasets/n2c2_train_vocab.spacy")
    print("DONE")

    ##DEBUG PRINTS -- Weird stuff with spans...

    # print(len(doc))
    # print("here they are:"+doc[4611].text+doc[4612].text+doc[4613].text)
    # print(len(doc.user_data["subwords"]))
    # exit()
    # i=0
    # print(len(doc))
    # print(len(doc.user_data["spans"]))
    # print(len(doc.user_data["subword_spans"]))
    # for token in doc:
    # print("Word: "+token.text+"\tSubwords: "+str(doc.user_data["subwords"][i])+"\t Word Spans: "+str(doc.user_data["spans"][i])+"\t Subword spans: "+ str(doc.user_data["subword_spans"][i]))
    # i+=1

##TODO: modify for use on multiple docs... after pipeline is fixed
