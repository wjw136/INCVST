import glob
import os
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import string
import argparse
from tqdm import tqdm
import copy
from collections import Counter
import random
import math
from os import listdir
from os.path import isfile, join, exists
import json
import copy
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to('cuda')
model.eval()
stop_words = set(['than', 'to', 'for', 'about', 'same', 'by', 'where',
                  'been', 'being', 'mightn', "shan't", 'wouldn', 'me', 'us', 'yours', 'you', 'he',
                  'here', "she's", 'she', 'i', "mustn't", 'y', 'our', 'those', 'haven', 'too',
                  'don', 'because', "won't", 'on', 'against', 'has', 'as', 'doing', "that'll",
                  'below', 'how', 'up', 'they', 'won', 'that', "aren't", 'some', 'so', 'theirs',
                  'didn', 'should', 'weren', 'having', 'an', 'nor', "needn't", 'of', 'yourselves',
                  'had', 'then', 'from', 'myself', 'few', "weren't", 'ours', 'couldn', 'will',
                  'needn', 'doesn', 'whom', 'themselves', "didn't", 'more', 'yourself', 'after',
                  'ain', 'are', 'does', "hasn't", 'ma', 'have', 'but', 'who', 'were', 'out',
                  'not', 'only', 'very', 'd', 'hers', 'what', 'my', 'and', "isn't", 'is',
                  'until', 'such', 'or', 't', 's', 'do', 'while', "you'll", 'it', 'their', 'am',
                  'was', 'be', 'shan', "couldn't", 'over', 'its', 'in', 'these', 've', "doesn't",
                  'we', 'can', 'hadn', 'his', "it's", 'other', 're', 'at', 'you', 'this', 'hasn',
                  'the', 'further', 'both', "you'd", 'your', "should've", 'a', 'any', 'why',
                  "shouldn't", "haven't", 'isn', 'her', "you're", 'again', "wasn't", 'did', "hadn't",
                  'own', "mightn't", 'down', 'herself', 'o', 'aren', 'shouldn', 'him', 'once', 'there',
                  'most', 'mustn', 'off', 'ourselves', 'each', 'above', 'now', 'before', 'with', 'under',
                  "don't", "wouldn't", 'which', 'if', 'when', 'himself', 'wasn', 'all', 'just', 'through',
                  'them', 'between', 'would', 'she', 'm', 'during', 'no', 'itself', "you've", 'into', 'll', '.', '...',
                  '?', '!',
                  'something', 'everything', 'nothing'])


def bertfortype(context:str,index:int,entity:str,top_k:int):
    # top_k = 1

    # context = "input sentence"

    text1 = '[CLS] ' + context + " {} is a [MASK].".format(entity) +'[SEP]'
    text2= '[CLS]'+context+ ' {} is a [MASK] party.'.format(entity)+'[SEP]'
    text3='[CLS]'+context+' {} is a [MASK] branch.'.format(entity)+'[SEP]'
    text4='[CLS]'+ context#+' {} is [MASK] of {}.'.format(entity[0],entity[1])+'[SEP]'
    text5='[CLS]'+ context+' {} is a [MASK].'.format(entity)+'[SEP]'
    text6='[CLS]'+context+'What is the property name of <subj>'

    # text4='[CLS]'+context+' {} is a [MASK] branch.'.format(entity)+'[SEP]'

    altext=[text1,text2,text3,text4,text5]

    tokenized_text = tokenizer.tokenize(altext[index])
    # print(altext[index])

    masked_index = tokenized_text.index('[MASK]')

    tokens_index = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([tokens_index])
    tokens_tensor = tokens_tensor.to('cuda')

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

        predictions = predictions.detach().cpu()
        predicted_index = np.argsort(predictions[0, masked_index])

        predicted_token = tokenizer.convert_ids_to_tokens(predicted_index[-10:])
        predicted_token = list(predicted_token)
        predicted_token.reverse()

        preserve_token = []
        for t in predicted_token:
            if t not in stop_words:
                preserve_token.append(t)

    return preserve_token[0]


cache_dir = "/root/jwwang/model"
tokenizer2 = T5Tokenizer.from_pretrained(join(cache_dir, 't5-3b'))
model2 = T5ForConditionalGeneration.from_pretrained(join(cache_dir, 't5-3b'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2.to(device)


def qafortype(context:str, q:str):
    first_prefix = 'question: '
    second_prefix = 'context: '
    input=first_prefix + q + ' ' + second_prefix + context
    batch_inputs = tokenizer2(input, return_tensors="pt", padding=True)
    output_sequences = model2.generate(
        input_ids=batch_inputs['input_ids'].to(device),
        attention_mask=batch_inputs['attention_mask'].to(device),
        do_sample=False,  # disable sampling to test if batching affects output
    )
    output_sequences = output_sequences.detach().cpu()
    batch_outputs = tokenizer2.batch_decode(output_sequences, skip_special_tokens=True)
    return batch_outputs

# print(
#     qafortype('The property , including Drummers \' Building and sidewalk , was added to the National Register of Historic Places on June 17 , 1982 , as " Jaeckel Hotel " .','where is jaeckel hotel listed on?')\
#     )