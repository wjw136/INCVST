import pickle
import pandas as pd
import numpy as np
import json
from os.path import join
import os
import regex as re
import math
import argparse

def qa_input(args):
	file_name = args.file_name
	model_name = args.model_name
	sents_batch_size = args.sents_batch_size
	output_data_dir = args.output_data_dir
	input_data_dir = join(output_data_dir,'qnli_output_stand_{}_{}'.format(file_name,model_name)) # which is the output dir of qnil_T5.py
	data_dir = args.data_dir
	input_data_path=args.input_data_path
	

	if not os.path.exists(input_data_dir):
		print("Error, {} does not exist".format(input_data_dir))
		raise SystemExit

	with open(join(input_data_path), "rb") as f:
		data = pickle.load(f)
	
	num_subfiles = math.ceil(len(data["rel"])/sents_batch_size)
	

	with open(join(data_dir,'q2id.pkl') ,mode='rb') as fin:
		q2id = pickle.load(fin)
		id2q = {}
		for key in q2id.keys():
			id2q[q2id[key]] = key

	with open(join(data_dir,'rel_q.pkl') ,mode='rb') as fin:
		rel2q = pickle.load(fin)
		qid2rel = {}
		for rel, q_list, q_id_list in zip(rel2q['rel'],rel2q['question'],rel2q['question_id']):
			for q_id in q_id_list:
				qid2rel[q_id] = rel

	qa_input = {"text":[],"text_id":[],"question":[],"question_id":[],"subj":[],"obj":[],"rel":[]}


	for i in range(0,num_subfiles):
		if not os.path.exists(join(input_data_dir,'o_'+file_name+'_'+str(i)+'.pkl')):
			print('WARNING: {} does not exist'.format(join(input_data_dir,'o_'+file_name+'_'+str(i)+'.pkl')))
			continue
		with open(join(input_data_dir,'o_'+file_name+'_'+str(i)+'.pkl'),mode='rb') as fin:
			output=pickle.load(fin)
			print("INFO: There are {} questions in {}".format(len(output),'o_'+file_name+'_'+str(i)+'.pkl'))

		for text_id, q_id, result in output:
			q_template = id2q[q_id]
			# print(text_id)
			# print(len(data['subj']))
			q_text = re.sub(r'<subj>', data['subj'][text_id], q_template)
			# print(q_text)
			q_text = re.sub(r'<obj>', data['obj'][text_id], q_text)
			qa_input["text"].append(data['text'][text_id]) 
			qa_input["text_id"].append(text_id) 
			qa_input["question"].append(q_text)
			qa_input["question_id"].append(q_id)
			qa_input["subj"].append(data["subj"][text_id])
			qa_input["obj"].append(data["obj"][text_id])
			qa_input["rel"].append(data["rel"][text_id]) 
			# print('TEXT     : ',data['text'][text_id])
			# print('Question : ',q_template)
			# print('TRUTH    : ',data['rel'][text_id])
			# print('PREDICT  : ',qid2rel[q_id])
			# print('\n')

	with open(join(output_data_dir,"qa_input_standford_{}.pkl".format(file_name)),mode='wb') as fo:
		pickle.dump(qa_input,fo)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--file_name', type=str, default='small_train',
	                    help='name of input data')
	parser.add_argument('--model_name', type=str, default='t5-3b',
	                    help='name of pretrain T5 model')
	parser.add_argument('--sents_batch_size',type=int, default=100,
						help='sentences batch size for storing results')
	parser.add_argument('--output_data_dir',type=str, default='/data/zzengae/inferT5/data/tacred',
						help='data directory')
	parser.add_argument('--data_dir',type=str, default='data/tacred',
						help='data directory')
	parser.add_argument('--input_data_path',type=str,default='data/',
						help='input data')
	
	args = parser.parse_args()

	qa_input(args)
	