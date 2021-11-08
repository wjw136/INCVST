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

def qa_infer(args):
	
	file_name = args.file_name
	batch_size = args.batch_size
	model_name = args.model_name
	# data_dir = args.data_dir
	use_gpu = args.use_gpu
	out_data_dir=args.out_data_dir

	first_prefix = 'question: '
	second_prefix = 'context: '

	
	cache_dir = "/root/jwwang/model"




	tokenizer = T5Tokenizer.from_pretrained(join(cache_dir,model_name))
	model = T5ForConditionalGeneration.from_pretrained(join(cache_dir,model_name))

	tokenizer.padding_side = "left"
	tokenizer.pad_token = tokenizer.eos_token # to avoid an error
	
	if use_gpu:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.to(device)
	
	output_dir = join(out_data_dir,'qa_output_standford_{}_{}'.format(file_name,model_name))
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

	with open(join(out_data_dir,'qa_input_standford_{}.pkl'.format(file_name)),mode='rb') as fin:
		data = pickle.load(fin)
	
	num_of_samples = len(data["text_id"])

	num_batch = math.ceil(num_of_samples / batch_size)

	text_q_result = []

	with torch.no_grad():
		for i in tqdm(range(num_batch)):
			batch_inputs = []
			batch_text_q_ids = []
			for text, text_id, q, q_id in zip(data["text"][i*batch_size:(i+1)*batch_size],\
				data["text_id"][i*batch_size:(i+1)*batch_size],\
				data['question'][i*batch_size:(i+1)*batch_size],\
				data['question_id'][i*batch_size:(i+1)*batch_size]):
				batch_inputs.append(first_prefix+q+' '+second_prefix+text)
				batch_text_q_ids.append([text_id,q_id])
			num_of_samples_per_batch = len(batch_inputs)
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
			
			for text_q, out in zip(batch_text_q_ids,batch_outputs):
				text, que = text_q
				text_q_result.append([text, que, out])

			if args.verbose:
				for j in range(num_of_samples_per_batch):
					text_id = i*batch_size+j
					print('TEXT       : ',data['text'][text_id])
					print('Question   : ',data['question'][text_id])
					print('TRUTH REL  : ',data['rel'][text_id])
					print('TRUTH OBJ  : ',data['obj'][text_id])
					print('PREDICT    : ',batch_outputs[j])
					print('\n')

		with open(join(output_dir,'o_'+file_name+'.pkl'), mode='wb') as fo:
			pickle.dump(text_q_result,fo)


			


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='small_train',
                        help='name of input data')
    parser.add_argument('--model_name', type=str, default='t5-3b',
                        help='name of pretrain T5 model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu or not')
    parser.add_argument('--verbose', action='store_true',
                        help='use gpu or not')
    parser.add_argument('--data_dir',type=str, default='/data/zzengae/inferT5/data/tacred',
                        help='data directory')
    parser.add_argument('--out_data_dir', type=str, default='/data/zzengae/inferT5/data/tacred',
                        help='data directory')
    args = parser.parse_args()

    time_start = time.time()
    qa_infer(args)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
