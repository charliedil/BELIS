import os
import pickle
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from spacy.vocab import Vocab
from transformers import AutoTokenizer, AutoModel

#Method for tokenizing txt files within path provided
def tokenize(path):
    nlp = spacy.load("en_core_web_sm")
    doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True) #where we will append the doc files to.
    all_files = list(os.listdir(path))
    txt_files = []
    for file in all_files:
        if file.endswith(".txt"):
            txt_files.append(file)
    for file in tqdm(txt_files, "Documents parsed"):
        if file != "102365.txt" and file != "117745.txt" and file != "120301.txt":
            print(file)
            text = ""
            with open(path + file, "r") as f:
                text = f.read()
            doc = nlp(text) #pass document text through spacy pipeline
            i = 0
            spacy_tok_sents = []
            sent = []
            first = True
            for token in doc: ##collect the sentences
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
            if len(sent) != 0: #with the way the code is written, we miss the last sentence, so i had to add this in
                spacy_tok_sents.append(sent)

            ##BERT tokenizer time
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            bert_offsets = tokenizer(spacy_tok_sents, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt", is_split_into_words=True,
                                     return_offsets_mapping=True).offset_mapping

            bert_tokens = tokenizer(spacy_tok_sents, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt", is_split_into_words=True)
            model.eval()
            # bert_outputs = model(**bert_tokens).last_hidden_state.cpu().numpy()
            bert_outputs = model(**bert_tokens).last_hidden_state.detach()
            doc.user_data = {}
            doc.user_data["filename"] = file
            doc.user_data["subwords"] = [] #subwords grouped by the word they are a part of
            doc.user_data["subword_embeddings"] = [] #embeddings of the subwords, same grouping as previous line
            doc.user_data["spans"] = [] #spans of the words
            doc.user_data["subword_spans"] = [] #spans of the subwords, grouped in same way as embeddings
            i = 1
            j = 0
            # for more than one doc, for doc in docs?
            subwords = [] #temporary array for holding subwords.
            subword_embeddings = [] #temporary array for holding subword embeddings
            subword_spans = [] #temporary array for holding subword_spans
            first = True #idk what this is tbh
            finished = True #is the word finished?
            prev = "" #keep track of previous "token"
            tcounter = 0

            for token in doc: #we compare the subtokens to this token
                doc.user_data["spans"].append((token.idx, len(token.text) + token.idx)) #span of the whole word
                while (j < bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102): #when we reach the end of a sentence, move up until a new sentence
                    i = 1
                    j += 1
                    # bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                if j == bert_tokens.input_ids.size()[0]: ##we've reached the end of the document. append what we have saved and move on.
                    doc.user_data["subwords"].append(subwords)
                    doc.user_data["subword_embeddings"].append(subword_embeddings)
                    doc.user_data["subword_spans"].append(subword_spans)
                    break

                if token.text.lower().startswith(
                        tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) and finished == True:#start of new word?
                    if token.text.lower() != tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).lower():#matches only the beginning
                        finished = False
                    else: #matched the whole word, which means we are good to move on.
                        finished = True
                    if first == False: #beginning of new word, so whatever we calculated previously must be done.
                        doc.user_data["subwords"].append(subwords)
                        subwords = []
                        doc.user_data["subword_embeddings"].append(subword_embeddings)
                        subword_embeddings = []
                        doc.user_data["subword_spans"].append(subword_spans)
                        subword_spans = []
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) ##add on what we have so far
                    subword_embeddings.append(bert_outputs[j][i].cpu().numpy())
                    subword_spans.append((int(bert_offsets[j][i][0]) + token.idx,
                                          int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + token.idx))
                    ##MISSING i+=1??? - no. there is one at the end
                elif tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith( ##is it the next subword?
                        "##") or finished == False:
                    subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                    subword_embeddings.append(bert_outputs[j][i].cpu().numpy())
                    if tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##"): ##account for the ##s if necessary in spans
                        subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx - 3))
                    else:
                        subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx))
                    i += 1## move to next subword
                    if i < bert_tokens.input_ids[j].size()[0]: ##make sure it is in bounds

                        #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                        temp = "" + subwords[0].lower() + subwords[1].lower() #combining existing subwords
                        if temp.replace("##", "").lower() == prev.lower(): #are we finished?
                            finished = True
                        while (tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith(
                                           "##") or finished == False):# loop until we finish the word, then we can move on
                            subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))) ##add the subword
                            subword_embeddings.append(bert_outputs[j][i].cpu().numpy()) #add teh embedding
                            if tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])).startswith("##"): #accounting for the ##s in spans
                                subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                                  int(bert_offsets[j][i][1]) + int(
                                                      bert_offsets[j][i][0]) + prev_idx - 3))
                            else:
                                subword_spans.append((int(bert_offsets[j][i][0]) + prev_idx,
                                                  int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + prev_idx))
                            i += 1#next subword
                            #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                            temp += subwords[len(subwords) - 1].lower()
                            if prev.lower() == temp or prev.lower() == temp.lower().replace("##", ""): ##checking if we're done
                                finished = True
                        finished = True #redundant

                        while (j < bert_tokens.input_ids.size()[0] and int(bert_tokens.input_ids[j][i]) == 102): #are we at the end of the sentence
                            i = 1
                            j += 1
                            #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                        if j == bert_tokens.input_ids.size()[
                            0]:  ##we've reached the end of the document. append what we have saved and move on.
                            doc.user_data["subwords"].append(subwords)
                            doc.user_data["subword_embeddings"].append(subword_embeddings)
                            doc.user_data["subword_spans"].append(subword_spans)
                            break
                        if token.text.lower().startswith(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))): #catch up to current token
                            if token.text.lower() != tokenizer.convert_ids_to_tokens(
                                int(bert_tokens.input_ids[j][i])).lower():#begins but not complete the word
                                finished = False
                            else:
                                finished = True #subword is the word, so we can move on
                            if first == False: #this may be redundant - always true
                                doc.user_data["subwords"].append(subwords)
                                doc.user_data["subword_embeddings"].append(subword_embeddings)
                                doc.user_data["subword_spans"].append(subword_spans)
                                subwords = []
                                subword_embeddings = []
                                subword_spans = []
                            subwords.append(tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i])))
                            subword_embeddings.append(bert_outputs[j][i].cpu().numpy())
                            subword_spans.append((int(bert_offsets[j][i][0]) + token.idx,
                                              int(bert_offsets[j][i][1]) + int(bert_offsets[j][i][0]) + token.idx))
                        else:
                            doc.user_data["subwords"].append(subwords)
                            doc.user_data["subword_embeddings"].append(subword_embeddings)
                            doc.user_data["subword_spans"].append(subword_spans)
                            subwords = []
                            subword_embeddings = []
                            subword_spans = []

                            i -= 1 #whitespace, skip
                            #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                else: #HERE
                    doc.user_data["subwords"].append(subwords)
                    doc.user_data["subword_embeddings"].append(subword_embeddings)
                    doc.user_data["subword_spans"].append(subword_spans)
                    subwords = []
                    subword_embeddings = []
                    subword_spans = []
                    finished = True #redundant?
                    i -= 1 #whitespace, skip
                    #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                i += 1 #move to next token
                #bert_token = tokenizer.convert_ids_to_tokens(int(bert_tokens.input_ids[j][i]))
                first = False
                prev = token.text #save previous in case not finished
                prev_idx = token.idx #save previous index
            ##END LOOP
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
                            end_span = int(l.split("\t")[1].split(" ")[len(l.split("\t")[1].split(" "))-1])
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
