from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
import pickle
from tqdm import tqdm
from os.path import join
import json
import os
import re
import torch
import argparse
import numpy as np
import time

def qnli_infer(args):
    file_name = args.file_name
    batch_size = args.batch_size
    model_name = args.model_name
    output_data_dir = args.output_data_dir
    use_gpu = args.use_gpu
    sents_batch_size = args.sents_batch_size # maximum samples being read in the program, due to memory limit, each time we only read a small number of sentences.
    data_dir = args.data_dir
    input_data_dir=args.input_data_dir
    flag=args.flag

    first_prefix = 'qnli question: '
    second_prefix = 'sentence: '

    cache_dir = "/root/jwwang/model"
    # cache_dir='/root/data/zzengae/transformers'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = join(output_data_dir,'qnli_output_stand_{}_{}_{}'.format(file_name,model_name,flag))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(join(input_data_dir), "rb") as f:
        data = pickle.load(f)
        sentences = data["text"]
        sentences_id=data['textid']
        sentence_subjs = data["subj"]
        sentence_objs = data["obj"]
        # wiki
        sentence_subj_types = data["subj_type"]
        sentence_obj_types = data["obj_type"]


    id2q = {}
    with open(join(data_dir,'q2id.pkl'), "rb") as f:
        q2id = pickle.load(f)
        questions = q2id.keys()
        for key in q2id.keys():
            id2q[q2id[key]] = key



    q2rel = {}
    with open(join(data_dir,'q2rel.pkl'), "rb") as f:
        q2rel = pickle.load(f)

    rel_type={}
    with open(join('data/output/rel2q.pkl'),'rb') as f:
        rel_type=pickle.load(f)


    tokenizer = T5Tokenizer.from_pretrained(join(cache_dir,model_name))
    model = T5ForConditionalGeneration.from_pretrained(join(cache_dir,model_name))
    print('model loading end'+'*'*50)

    if use_gpu:
        model.to(device)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    num_of_samples = len(sentences) * len(questions)

    # num_batch = math.ceil(num_of_samples / batch_size)

    i_num_batch = math.ceil(len(sentences)/sents_batch_size)

    actual_num_batch = 0

    start_i = 0
    end_i = i_num_batch
    #10
    print(i_num_batch)
    with torch.no_grad():
        for i in tqdm(range(start_i,end_i)):
            sents_id=sentences_id[i*sents_batch_size:(i+1)*sents_batch_size]
            sents = sentences[i*sents_batch_size:(i+1)*sents_batch_size]
            subjs = sentence_subjs[i*sents_batch_size:(i+1)*sents_batch_size]
            objs = sentence_objs[i*sents_batch_size:(i+1)*sents_batch_size]
            # wiki
            subj_types = sentence_subj_types[i*sents_batch_size:(i+1)*sents_batch_size]
            obj_types = sentence_obj_types[i*sents_batch_size:(i+1)*sents_batch_size]
            inputs = []
            sent_q_id_inputs = []
            sent_q_result = []
            for s_id, s in enumerate(sents):
                # wiki
                subj_type = subj_types[s_id]
                obj_type = obj_types[s_id]
                for template_q in questions:
                    valid = True
                    # wiki
                    if "<subj>" in template_q:
                        # if q2rel[template_q][:len("per")] == "per":
                        # 	if subj_type != 'PERSON':
                        # 		valid = False
                        # elif q2rel[template_q][:len("org")] == "org":
                        # 	if subj_type != 'ORGANIZATION':
                        # 		valid = False
                        tmp_rel=q2rel[template_q]
                        if subj_type not in rel_type[tmp_rel]['subj'] or obj_type not in rel_type[tmp_rel]['obj']:
                            valid = False
                    elif "<obj>" in template_q:
                        # not implement
                        tmp_rel=q2rel[template_q]
                        if obj_type not in rel_type[tmp_rel]['obj'] or subj_type not in rel_type[tmp_rel]['subj']:
                            valid = False
                        pass

                    valid=True

                    if valid:
                        q = re.sub(r'<subj>', subjs[s_id], template_q)
                        q = re.sub(r'<obj>', objs[s_id], q)
                        inputs.append(first_prefix+q+' '+second_prefix+s) # substitute subj to template_q
                        sent_q_id_inputs.append([i*sents_batch_size+s_id,q2id[template_q]]) # i*sents_batch_size+s_id is global sentence id
                        # template_q_unk = re.sub(r'<subj>', 'unk', template_q)
                        # template_q_unk = re.sub(r'<obj>', 'unk', template_q_unk)
                        # inputs.append(first_prefix+template_q_unk+' '+second_prefix+s) # substitute subj to template_q

            j_num_batch = math.ceil(len(inputs)/batch_size)
            for j in tqdm(range(j_num_batch)):
                batch_inputs = inputs[j*batch_size:(j+1)*batch_size]
                batch_sent_q_id_inputs = sent_q_id_inputs[j*batch_size:(j+1)*batch_size]
                batch_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True)
                if use_gpu:
                    output_sequences = model.generate(
                        input_ids=batch_inputs['input_ids'].to(device),
                        attention_mask=batch_inputs['attention_mask'].to(device),
                        do_sample=False, # disable sampling to test if batching affects output
                    )
                    output_sequences = output_sequences.detach().cpu()
                    batch_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                else:
                    output_sequences = model.generate(
                        input_ids=batch_inputs['input_ids'],
                        attention_mask=batch_inputs['attention_mask'],
                        do_sample=False, # disable sampling to test if batching affects output
                    )
                    batch_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                actual_num_batch += 1


                for sent_q, out in zip(batch_sent_q_id_inputs,batch_outputs):
                    sent, que = sent_q
                    if out == 'entailment':
                        sent_q_result.append([sent, que, out])

            with open(join(output_dir,'o_'+file_name+'_'+str(i)+'.pkl'), mode='wb') as fo:
                pickle.dump(sent_q_result,fo)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, default='small_train',
                        help='name of input data')
    parser.add_argument('--flag', type=str, default='experiment diffs',
                        help='name of experiment')
    parser.add_argument('--model_name', type=str, default='t5-3b',
                        help='name of pretrain T5 model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu or not')
    parser.add_argument('--output_data_dir',type=str, default='/data/zzengae/inferT5/data/wiki80',
                        help='data directory') # output directory
    parser.add_argument('--data_dir',type=str, default='data/wiki80',
                        help='data directory') # directory to 'rel_q.pkl' 'q2id.pkl' 'q2rel.pkl'  'train.pkl' 'dev.pkl' 'test.pkl'
    parser.add_argument('--sents_batch_size',type=int, default=100,
                        help='sentences batch size for storing results')
    parser.add_argument('--input_data_dir',type=str, default='data/output/type_infer_output_standford_wiki80_train_t5-base',
                        help='input=>data.pkl')

    args = parser.parse_args()

    time_start = time.time()
    qnli_infer(args)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')