import argparse
import pandas as pd
import csv
import math
import numpy as np
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score
import torch.nn.functional as F

import torch
import os
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, RobertaTokenizer
from transformers import RobertaForTokenClassification, BertForTokenClassification, AdamW




parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, help="dir of the data (train, test, dev.test)")
parser.add_argument('-n', '--dist_gpu', help="set if number of GPUs is more than one", action='store_true')
parser.add_argument("-l", "--max_length", type=str, help="maximum lenght of sequence")
parser.add_argument("-b", "--batch_number", type=str, help="batch number in a GPU")
parser.add_argument("-e", "--epochs_number", type=str, help="number of epochs")


args = parser.parse_args()
data = args.data
distrubuted_training = args.dist_gpu
max_len = args.max_length
batch_num = args.batch_number
epochs = args.epochs_number

# Fillna method can make same sentence with same sentence name
def df_file(path):
    df_train = pd.read_csv(path + "train.tsv", sep='\t', header= None, quoting=csv.QUOTE_NONE, names= ['word', 'tag']).fillna(method='ffill')
    df_test =  pd.read_csv(path + "test.tsv", sep='\t', header= None, quoting=csv.QUOTE_NONE, names= ['word', 'tag']).fillna(method='ffill')
    df_dev =  pd.read_csv(path + "dev.tsv", sep='\t', quoting=csv.QUOTE_NONE, header= None, names= ['word', 'tag']).fillna(method='ffill')
    df = df_train.append([df_test, df_dev], ignore_index = True)
    return df


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = self.add_column(data)
        self.empty = False
        self.c = 1
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.data.groupby('sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def add_column(self, file):
        words = file.word.values.tolist()
        sents = []
        c = 1
        for i in words:
            if i == ".":
                c += 1
                sents.append(f"sentence_{c}")
            else:
                sents.append(f"sentence_{c}")
        file['sentence #'] = sents
        return file

df = df_file(data)
getter = SentenceGetter(df)

sentences = [[s[0] for s in sent] for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
tags_vals = list(set(df["tag"].values))


tags_vals.append('X')
tags_vals.append('[CLS]')
tags_vals.append('[SEP]')
tags_vals = set(tags_vals)

tag2idx = {t: i for i, t in enumerate(tags_vals)}
tag2name={tag2idx[key] : key for key in tag2idx.keys()}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu =  torch.cude.device_count()
#torch.cuda.device_count()
tok_dir = "/GW/Health-Corpus/work/roberta-finetuning-ner/roberta-tokenizer/roberta-base-"

tokenizer = RobertaTokenizer(tok_dir+"vocab.json", tok_dir + "merges.txt",do_lower_case=False)

# %%
tokenized_texts = []
word_piece_labels = []
i_inc = 0
for word_list, label in (zip(sentences, labels)):
    temp_lable = []
    temp_token = []

    # Add [CLS] at the front
    temp_lable.append('[CLS]')
    temp_token.append('[CLS]')

    for word, lab in zip(word_list, label):
        token_list = tokenizer.tokenize(word)
        for m, token in enumerate(token_list):
            temp_token.append(token)
            if m == 0:
                temp_lable.append(lab)
            else:
                temp_lable.append('X')

                # Add [SEP] at the end
    temp_lable.append('[SEP]')
    temp_token.append('[SEP]')

    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)


    i_inc += 1

# %%

# Make text token into id
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")
# Make label into id, pad with "O" meaning others
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                     maxlen=max_len, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
# For fine tune of predict, with token mask is 1,pad token is 0
attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
segment_ids = [[0] * len(input_id) for input_id in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks, tr_segs, val_segs = train_test_split(input_ids, tags,
                                                                                                    attention_masks,
                                                                                                    segment_ids,
                                                                                                    random_state=4,
                                                                                                    test_size=0.3)
# %%

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)

# %%


# Set batch num
# Only set token embedding, attention embedding, no segment embedding
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num, drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)
# %%
model_file_address = 'bert-base-cased'
model = BertForTokenClassification.from_pretrained(model_file_address, num_labels=len(tag2idx)).to(device)

if n_gpu > 1 and distrubuted_training:
    model = torch.nn.DataParallel(model)
max_grad_norm = 1.0
num_train_optimization_steps = int(math.ceil(len(tr_inputs) / batch_num) / 1) * epochs
# False: only fine tuning the classifier layers
FULL_FINETUNING = True
if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
# %%

model.train();
print("***** Running training *****")
print("  Num examples = %d" % (len(tr_inputs)))
print("  Batch size = %d" % (batch_num))
print("  Num steps = %d" % (num_train_optimization_steps))
for _ in trange(epochs, desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)
        loss, scores = outputs[:2]
        if n_gpu > 1:
            # When multi gpu, average it
            loss = loss.mean()

        # backward pass
        loss.backward()

        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

    # print train loss per epoch
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

bert_out_address = 'models/'
if not os.path.exists(bert_out_address):
    os.makedirs(bert_out_address)
# Save a trained model, configuration and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(bert_out_address, "pytorch_model.bin")
output_config_file = os.path.join(bert_out_address, "config.json")
# Save model into file
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(bert_out_address)

# %%
model = RobertaForTokenClassification.from_pretrained(bert_out_address, num_labels=len(tag2idx))
# Set model to GPU
model.cuda();
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
# Evalue loop
model.eval();
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
y_true = []
y_pred = []

print("***** Running evaluation *****")
print("  Num examples ={}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, label_ids = batch

    #     if step > 2:
    #         break

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
                        attention_mask=input_mask, )
        # For eval mode, the first result of outputs is logits
        logits = outputs[0]

        # Get NER predict result
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()

    # Get NER true result
    label_ids = label_ids.to('cpu').numpy()

    # Only predict the real word, mark=0, will not calculate
    input_mask = input_mask.to('cpu').numpy()

    # Compare the valuable predict result
    for i, mask in enumerate(input_mask):
        # Real one
        temp_1 = []
        # Predict one
        temp_2 = []

        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare
            if m:
                if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[
                    label_ids[i][j]] != "[SEP]":  # Exclude the X label
                    temp_1.append(tag2name[label_ids[i][j]])
                    temp_2.append(tag2name[logits[i][j]])
            else:
                break

        y_true.append(temp_1)
        y_pred.append(temp_2)

print("f1 socre: %f" % (f1_score(y_true, y_pred)))
print("Accuracy score: %f" % (accuracy_score(y_true, y_pred)))

# Get acc , recall, F1 result report
report = classification_report(y_true, y_pred, digits=4)

# Save the report into file
output_eval_file = os.path.join(bert_out_address, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    print("\n%s" % (report))
    print("f1 socre: %f" % (f1_score(y_true, y_pred)))
    print("Accuracy score: %f" % (accuracy_score(y_true, y_pred)))

    writer.write("f1 socre:\n")
    writer.write(str(f1_score(y_true, y_pred)))
    writer.write("\n\nAccuracy score:\n")
    writer.write(str(accuracy_score(y_true, y_pred)))
    writer.write("\n\n")
    writer.write(report)
