import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
import pickle
from tqdm import trange
from os.path import join
import json
import os
import re
import torch
import argparse
import numpy as np
from collections import Counter
from Levenshtein import *
from finegrainedbert import bertfortype,qafortype
from tqdm import tqdm

def plot_confusion_matrix(cm, labels_name, title):
    # print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # print('aaaaaaaaaaaaaa')
    plt.savefig('res.png')

def str_filter(token:str):
    token=token.lower()
    token=token.replace(' - ','-')
    token=token.replace('the ','')
    # token=token.replace(' ','')
    token = token.replace('\' ', '')
    token = token.replace(' \'', '')
    return token

# print(str_filter('The Bread of Those Early Years'))


#or str(pred.lower()).__contains__(gold.lower()) or distance(gold.lower(),pred.lower())<5:
#pred包含gold
def answer_correct(gold, pred,multiple:bool):
    gold=str_filter(gold)
    pred=str_filter(pred)
    if multiple:# and gold in pred:
        # print(pred)
        # pred_list=pred.split(',|and')
        # pred_list=[i.strip() for i in pred_list]
        # if gold in pred_list:
        if gold in pred:
            return True
        else:
            return False
    elif gold==pred or distance(gold.lower(),pred.lower())<2:
        return True
    else:
        return False






save_res = {'text': [], 'textid': [], 'p_rel': [], 'subj': [], 'obj': [] \
    , 'subj_type': [], 'obj_type': [],'subj_pos':[],'obj_pos':[]}

save_false={'text':[],'subject':[],'subject_type':[],'object':[],'object_type':[],'real relation':[],'Question relation':[],'Question':[],'obj type of question':[]}

svae_no_solve={}
# , 'subj_start': [], \
#             'obj_start': [], 'subj_end': [], 'obj_end': []}
def save_result(tid:int,type:str,data):
    save_res['text'].append(data['text'][tid])
    save_res['textid'].append(data['textid'][tid])
    save_res['p_rel'].append(type)
    save_res['subj'].append(data['subj'][tid])
    save_res['obj'].append(data['obj'][tid])
    save_res['subj_type'].append(data['subj_type'][tid])
    # print(data['subj_type'][tid])
    save_res['obj_type'].append(data['obj_type'][tid])
    save_res['obj_pos'].append(data['obj_pos'][tid])
    save_res['subj_pos'].append(data['subj_pos'][tid])
    # print('ssss')
    # print(pos[data['text'][tid]][0])

def main(args):
    # dir=args.dirname
    input_data_dir=args.input_data_dir
    data_dir = args.data_dir
    out_data_dir = args.out_data_dir
    file_name = args.file_name
    model_name = args.model_name
    output_identifier = args.output_identifier
    #input_data_dir = join(out_data_dir, '{}_output_{}_{}'.format(output_identifier, file_name, model_name))
    # input_data_dir = join(out_data_dir, '{}_intput_{}'.format(output_identifier, file_name, ))

    with open(input_data_dir, "rb") as f:
        data = pickle.load(f)

    print(data.keys())
    id2q = {}
    with open(join(data_dir, 'q2id.pkl'), "rb") as f:
        q2id = pickle.load(f)
        for key in q2id.keys():
            id2q[q2id[key]] = key
    qid2rel = {}
    with open(join(data_dir, 'rel_q.pkl'), "rb") as f:
        rel_q_with_ids = pickle.load(f)
        for rel, rel_id, que_list, que_id_list in zip(rel_q_with_ids['rel'], rel_q_with_ids['rel_id'],
                                                      rel_q_with_ids['question'], rel_q_with_ids['question_id']):
            for que_id in que_id_list:
                qid2rel[que_id] = rel



    data_dir_=join('data/','output/qa_output_standford_wiki80_train_t5-3b')
    with open(join(data_dir_, 'o_' + file_name + '.pkl'), mode='rb') as fin:
        output = pickle.load(fin)

    rel_type = {}
    with open('data/output/rel2q.pkl', 'rb') as f:
        rel_type = pickle.load(f)

    # rel_=list(rel_)
    # rel_=['screenwriter','after a work by','country of origin','contains administrative territorial entity','licensed to broadcast to','participant','league','constellation','located in or next to body of water', 'location', 'religion', 'original language of film or TV show', 'headquarters location', 'sibling', 'notable work', 'located in the administrative territorial entity', 'competition class', 'child', 'publisher', 'participating team', 'work location', 'member of', 'located on terrain feature', 'follows', 'spouse', 'tributary', 'military branch', 'father', 'developer', 'member of political party', 'language of work or name', 'subsidiary', 'occupation', 'mountain range', 'director', 'instrument', 'mother', 'mouth of the watercourse', 'residence']


#----------------------------------------------------------------------------------------------------------------------

    predict=[]
    truth=[]


    predictions = {}
    predictions_l={}
    results={}
    res_type={}
    questions={}
    questions_l={}
    golds = {}
    num_no_relation = 0

    cnt_answer_wrong=0

    cnt = 0

    q_multiple=["What instruments does <subj> play?",\
                "What teams are occupants of <subj>?",\
                "What are operating systems of <subj>?","What are platforms of <obj>?","What are tributaries of <subj>?",\
                "Who are the winners of <subj>?",\
                "What seasons of <obj> are mentioned?","What are <subj>'s fields of work?"\
                ,"What are notable works of <subj>?","What sports does <subj> play?","Who are <subj>'s children?",\
                "What are in the constellation of \" <obj> \"?"]
    # ans = []
    # print(max(qid))
    # print(min(qid))
    # print(max(id2q.keys()))

    counter=Counter()


    """
    筛选回答正确的并且和问题的类型对应的
    q q_list
    """
    for text_id, q_id, result in output:
        # print(text_id)
        golds[text_id] = data['rel'][text_id]
        # if data['rel'][text_id]=='org:dissolved':
        #     ans.append(data['obj_type'][text_id])
        print_flag = False
        if data['rel'][text_id] != 'no_relation':
            # print(q_id)
            # print(id2q[q_id])

            # if data['rel'][text_id]=='part of':
            #     print('subj:    {}, type:   {}'.format(data['subj'][text_id],data['subj_type'][text_id]))
            #     print('obj:    {}, type:   {}'.format(data['obj'][text_id],data['subj_type'][text_id]))
            multiple=False
            if str(id2q[q_id]).split(' ')[0].lower()=='where' or id2q[q_id] in q_multiple:
                multiple=True
            if "<obj>" in id2q[q_id] and data['obj_type'][text_id] in rel_type[qid2rel[q_id]]['obj'] :
                #and data['subj_type'][text_id]!='OTHER' and data['obj_type'][text_id]!='OTHER'
                print_flag = answer_correct(data['subj'][text_id], result,multiple)
            if "<subj>" in id2q[q_id] and data['subj_type'][text_id] in rel_type[qid2rel[q_id]]['subj'] :
                #and data['subj_type'][text_id]!='OTHER' and data['obj_type'][text_id]!='OTHER'
                print_flag = answer_correct(data['obj'][text_id], result,multiple)
            if print_flag:
                if text_id not in predictions.keys():
                    predictions[text_id] = set() #set无序且去重
                    predictions_l[text_id]=[]
                    questions[text_id]=set()
                    questions_l[text_id] = []
                # results[text_id] = set()
                predictions[text_id].update([qid2rel[q_id]])
                predictions_l[text_id].append(qid2rel[q_id])
                # print(result)
                results[text_id]=result
                if "<obj>" in id2q[q_id]:
                    res_type[text_id] = data['subj_type'][text_id]
                if "<subj>" in id2q[q_id]:
                    res_type[text_id] = data['obj_type'][text_id]
                questions[text_id].update([id2q[q_id]])
                questions_l[text_id].append(id2q[q_id])
            elif qid2rel[q_id]==data['rel'][text_id]:

                #提的问题的关系和实际的关系相同
                cnt_answer_wrong+=1

                # cnt+=1
                # print('*'*20)
                # print('text:    {}'.format(data['text'][text_id]))
                # print("subject:    {}".format(data['subj'][text_id]))
                # print('subject type:    {}'.format(data['subj_type'][text_id]))
                # print("object:     {}".format(data['obj'][text_id]))
                # print("object type:     {}".format(data['obj_type'][text_id]))
                # print('real relation:   {}'.format(data['rel'][text_id]))
                # print('question:    {}'.format(id2q[q_id]))
                # print('question relation:   {}'.format(qid2rel[q_id]))
                # print('ans:     {}'.format(result))
                # rel.update([data['rel'][text_id]])
                # predict.append(rel2id[data['rel'][text_id]])
                # # rel.update([qid2rel[q_id]])
                # truth.append(rel2id[qid2rel[q_id]])
        else:
            num_no_relation += 1

    # cnt_h=0
    # for key in predictions.keys():
    #     if len(predictions[key]) == 1:
    #         cnt_h+=1

    cnt_all=0
    cnt_true=0
    cnt_1=0

    cnt_2=0
    cnt_3=0



    """
    fine type for conflicts
    """
    #细化类别
    #todo
    for key in predictions.keys():
        if "language of work or name" in predictions[key] and 'original language of film or TV show' in predictions[key]:
            # cnt_1+=1
            res=bertfortype(data['text'][key],0,data['subj'][key],1)
            list2=['flim','moive','serial','drama']
            list1= ['book','song','novel','poem','newspaper','magazine','name','surname','pseudonym']
            for i in res:
                # if data['rel'][key]=='language of work or name':
                #     print('-',end='')
                #     print(res)
                # elif data['rel'][key]=='original language of film or TV show':
                #     print(res)
                if i.lower().strip() in list1:
                    predictions[key]=set(["language of work or name"]) #set的初始化要用list
                    # print(predictions[key])
                    predictions_l[key]=["language of work or name"]
                    questions[key]=set(['fine grained question'])
                    questions_l[key]=['fine grained question']
                    # cnt_all += 1
                    # if data['rel'][key]=='language of work or name':
                    #     cnt_true+=1
                    # else:
                    #     print(i, end='   :   ')
                    #     print(data['rel'][key])
                    # break
                if i.lower().strip() in list2:
                    predictions[key] = set(["original language of film or TV show"])
                    predictions_l[key] = ["original language of film or TV show"]
                    questions[key] = set(['fine grained question'])
                    questions_l[key] = ['fine grained question']
                    # cnt_all += 1
                    # if data['rel'][key] == 'original language of film or TV show':
                    #     # cnt_true += 1
                    # else:
                    #     print(i,end='   :   ')
                    #     print(data['rel'][key])
                    # break
            # pass
        if len(set(['member of', 'member of political party', 'military branch']).intersection(set(predictions[key])))>1:
            res1=bertfortype(data['text'][key],1,data['obj'][key],1)
            res2 = bertfortype(data['text'][key], 2, data['obj'][key],1)
            flag=True
            for i,j in zip(res1,res2):
                if i.lower().strip()=='political' and j.lower().strip()!='military':
                    # print(res1)
                    predictions[key] = set(["member of political party"])
                    predictions_l[key] = ["member of political party"]
                    questions[key] = set(['fine grained question'])
                    questions_l[key] = ['fine grained question']
                    # cnt_all += 1
                    # if data['rel'][key] == 'member of political party':
                        # cnt_true += 1
                    # flag=False
                    break
                if j.lower().strip()=='military':
                    predictions[key] = set(["military branch"])
                    predictions_l[key] = ["military branch"]
                    questions[key] = set(['fine grained question'])
                    questions_l[key] = ['fine grained question']
                    # cnt_all += 1
                    # if data['rel'][key] == 'military branch':
                    #     cnt_true += 1
                    # flag=False
                    break
                # if j.lower().strip()!='military' and i.lower().strip()!='political':
                #     print(data['rel'][key])
            # if flag:
            #     pass
                # predictions[key] = set("member of")
                # predictions_l[key] = ["member of"]
                # questions[key] = set('fine grained question')
                # questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'member of':
                #     cnt_true += 1
            # pass
        if 'composer' in predictions[key] or 'screenwriter' in predictions[key]:
            # cnt_1+=1
            list1=["symphony","variation"]
            list2=["film","movie","novel"]
            res = bertfortype(data['text'][key], 0, data['subj'][key], 1)[0]
            if res.lower().strip() in list1:
                predictions[key] = set(["composer"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["composer"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'composer':
                #     cnt_true += 1
                # else:
                #     print(res, end='   :   ')
                #     print(data['rel'][key])
            if res.lower().strip() in list2:
                predictions[key] = set(["screenwriter"])
                predictions_l[key] = ["screenwriter"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'screenwriter':
                #     cnt_true += 1
                # else:
                #     print(res,end='   :   ')
                #     print(data['rel'][key])
        if len(set(['mountain range', 'mouth of the watercourse', 'located in or next to body of water','constellation']).intersection(set(predictions[key])))>0:
            # cnt_1+=1
            list1_0=["mountain","glacier"];list1_1=["mountain","glacier"]
            list2_0=["river","lake","stream","tributary"];list2_1=["river","lake","stream","tributary"]
            list3_0=["river","lake","stream"];list3_1=["river","lake","stream"]
            list4_0=["star,constellation"];list4_1=["star,constellation"]
            # list2=["film","movie","novel"]
            res_0 = bertfortype(data['text'][key], 0, data['subj'][key], 1)[0]
            res_1= bertfortype(data['text'][key], 0, data['obj'][key], 1)[0]

            if res_0.lower().strip() in list1_0 or res_1.lower().strip() in list1_1:
                predictions[key] = set(["mountain range"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["mountain range"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'mountain range':
                #     cnt_true += 1
                # else:
                #
                #     print('1')
                #     print(res_0, end='   :   ')
                #     print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list2_0 or res_1.lower().strip() in list2_1:
                predictions[key] = set(["mouth of the watercourse"])
                predictions_l[key] = ["mouth of the watercourse"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'mouth of the watercourse':
                #     cnt_true += 1
                # else:
                #     print('2')
                #     print(res_0,end='   :   ')
                #     print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list3_0 or res_1.lower().strip() in list3_1:
                predictions[key] = set(["located in or next to body of water"])
                predictions_l[key] = ["located in or next to body of water"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'located in or next to body of water':
                #     cnt_true += 1
                # else:
                #     print('3')
                #     print(res_0, end='   :   ')
                #     print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list4_0 or res_1.lower().strip() in list4_1:
                predictions[key] = set(["constellation"])
                predictions_l[key] = ["constellation"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'constellation':
                #     cnt_true += 1
                # else:
                #     print('4')
                #     print(res_0, end='   :   ')
                #     print(res_1, end='   :   ')
                #     print(data['rel'][key])
        if len(set(['developer', 'manufacturer', 'distributor','record label']).intersection(set(predictions[key])))>1:
            # cnt_1+=1
            list1_0=["game",'sequel',"website"];#list1_1=["mountain","glacier"]
            list2_0=["model"];#list2_1=["river","lake","stream","tributary"]
            list3_0=["film","moive"];#list3_1=["river","lake","stream"]
            list4_0=["song","band","rapper","musician"];#list4_1=["star,constellation"]
            # list2=["film","movie","novel"]
            res_0 = bertfortype(data['text'][key], 0, data['subj'][key], 1)[0]
            # res_1= bertfortype(data['text'][key], 0, data['obj'][key], 1)[0]

            if res_0.lower().strip() in list1_0 :#or res_1.lower().strip() in list1_1:
                predictions[key] = set(["developer"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["developer"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'developer':
                #     cnt_true += 1
                # else:
                #
                #     print('1')
                #     print(res_0, end='   :   ')
                #     # print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list2_0 :#or res_1.lower().strip() in list2_1:
                predictions[key] = set(["manufacturer"])
                predictions_l[key] = ["manufacturer"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'mouth of the watercourse':
                #     cnt_true += 1
                # else:
                #     print('2')
                #     print(res_0,end='   :   ')
                #     # print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list3_0 :#or res_1.lower().strip() in list3_1:
                predictions[key] = set(["distributor"])
                predictions_l[key] = ["distributor"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'distributor':
                #     cnt_true += 1
                # else:
                #     print('3')
                #     print(res_0, end='   :   ')
                #     # print(res_1, end='   :   ')
                #     print(data['rel'][key])
            if res_0.lower().strip() in list4_0 :#or res_1.lower().strip() in list4_1:
                predictions[key] = set(["record label"])
                predictions_l[key] = ["record label"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                # cnt_all += 1
                # if data['rel'][key] == 'record label':
                #     cnt_true += 1
                # else:
                #     print('4')
                #     print(res_0, end='   :   ')
                #     # print(res_1, end='   :   ')
                #     print(data['rel'][key])
        if 'headquarters location' in predictions[key]:
            list1=["company","conglomerate","subsidiary"]
            # print(data['text'][key])
            res= bertfortype(data['text'][key], 0, data['subj'][key], 1)[0]
            if res.lower().strip() in list1:
                predictions[key] = set(["headquarters location"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["headquarters location"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
        if 'original network' in predictions[key]:
            # print('aaaaa')
            list1=["broadcaster","network",'musical','documentary']
            res= bertfortype(data['text'][key], 0, data['subj'][key], 1)[0] #大小写--------------------------------------------------
            counter.update([res])
            if res.lower().strip() in list1:
                # print('bbb')
                predictions[key] = set(['original network'])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ['original network']
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
                cnt_all += 1
                if data['rel'][key] == 'original network':
                    cnt_true += 1
                else:
                    print(res, end='   :   ')
                    print(data['rel'][key])











    # for i,j in counter.items():
    #     print(i,end=':  ')
    #     print(j)
    # print(cnt_1)


    # print(cnt_2)



    """
    fine type for OTHER
    """
    for key in tqdm(predictions.keys()):
        # key=predictions.keys()[i]
        if len(list(predictions[key])) == 1 and (data['subj_type'][key]=='OTHER' or data['obj_type'][key]=='OTHER'):
            res_o = bertfortype(data['text'][key], 0, data['obj'][key], 1)
            res_s = bertfortype(data['text'][key], 0, data['subj'][key], 1)
            # print(res_s)
            if res_o =='soprano':
                predictions[key] = set(["voice type"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key]=["voice type"]
                questions[key]=set(['fine grained question'])
                questions_l[key]=['fine grained question']
            elif res_o =='specialty':
                predictions[key] = set(["instrument"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["instrument"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_o=='democrat':
                predictions[key] = set(["position held"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["position held"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_s=='municipality':
                predictions[key] = set(["country"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["country"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_o=='pseudonym':
                predictions[key] = set(["occupation"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["occupation"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_s=='documentary':
                predictions[key] = set(["participant"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["participant"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_s=='manga':
                predictions[key] = set(["characters"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["characters"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_o=='genre':
                predictions[key] = set(["genre"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["genre"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_o=='discipline':
                predictions[key] = set(["field of work"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["field of work"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            elif res_o=='midfielder':
                predictions[key] = set(["position played on team / speciality"])  # set的初始化要用list
                # print(predictions[key])
                predictions_l[key] = ["position played on team / speciality"]
                questions[key] = set(['fine grained question'])
                questions_l[key] = ['fine grained question']
            else:
                ans=qafortype(data['text'][key],'where is {} listed on?'.format(data['subj'][key]))
                if ans==data['obj'][key]:
                    predictions[key] = set(["heritage designation"])  # set的初始化要用list
                    # print(predictions[key])
                    predictions_l[key] = ["heritage designation"]
                    questions[key] = set(['fine grained question'])
                    questions_l[key] = ['fine grained question']
                else:
                    predictions[key] = set()
                    predictions_l[key] = []
                    questions[key] = set(['fine grained question'])
                    questions_l[key] = ['fine grained question']







    #         # elif data['subj_type'][key]=='OTHER' or data['obj_type'][key]=='OTHER':
    #         #     predictions[key]=set()
    #         #     predictions_l[key]=[]
    #         #     questions[key] = set(['fine grained question'])
    #         #     questions_l[key] = ['fine grained question']














    #         counter.update([res_o])
    # #         print('aaaa')
    # #
    # for i, j in counter.items():
    #     print(i)
    #     print(j)



    # for key in predictions.keys():
    #     if len(list(predictions[key]))==1 and list(predictions[key])[0] in ['owned by', 'original network']:
    #         if 'part' == bertfortype(data['text'][key],3,[data['subj'][key],data['obj'][key]],10):
    #             predictions[key] = set(["part of"])  # set的初始化要用list
    #             # print(predictions[key])
    #             predictions_l[key]=["part of"]
    #             questions[key]=set(['fine grained question'])
    #             questions_l[key]=['fine grained question']
    #         else:
    #             predictions[key] = set()  # set的初始化要用list
    #             # print(predictions[key])
    #             predictions_l[key] = []
    #             questions[key] = set()
    #             questions_l[key] = []
    # for key in predictions.keys():
    #

    # print('cnt:     {}, cnt_true:   {}'.format(cnt_all,cnt_true))
    # print('fine_grained end'+'*'*10)

    # cnt_j=0
    # for key in predictions.keys():
    #     if len(predictions[key]) == 1:
    #         cnt_j+=1
    # print(cnt_h)
    # print(cnt_j)

# ----------------------------------------------------------------------------------------------------------------------

    num_conflits = 0
    num_no_conflits = 0
    num_correct = 0
    label_diversity = Counter()

    # rel2type={}

    num_con_solved = 0
    num_con_nsolved = 0
    num_con_0 = 0
    num_solved_true = 0

    my_count=Counter()
    my_count_1=Counter()




    """
    类型对应+统计
    """
    # or 'UNKNOWN' in rel_type[j]['subj']
    for key in predictions.keys():
        s_type=data['subj_type'][key]
        o_type=data['obj_type'][key]
        flag_1=False
        for i,j in zip(questions_l[key],predictions_l[key]):
            if ("<obj>" in i and s_type in rel_type[j]['subj']) or i=='fine grained question':# or 'UNKNOWN' in rel_type[j]['subj']):
                flag_1=True
            if ("<subj>" in i and o_type in rel_type[j]['obj']) or i=='fine grained question':# or 'UNKNOWN' in rel_type[j]['subj']):
                flag_1=True
        if len(predictions[key]) == 1 and flag_1:
            num_no_conflits += 1
            save_result(key,list(predictions[key])[0],data)
            if list(predictions[key])[0] == golds[key]:
                num_correct += 1
                label_diversity.update([golds[key]])
            else:
                my_count.update([golds[key]])
                my_count.update([list(predictions[key])[0]])
                my_count_1.update([(golds[key],list(predictions[key])[0])])
                #todo <<false_rel>>

                # predict.append(rel2id[golds[key]])
                # rel.update([qid2rel[q_id]])
                # truth.append(rel2id[list(predictions[key])[0]])
                # cnt+=1
                # print('*'*20)
                # print('text:    {}'.format(data['text'][key]))
                # print("subject:    {}".format(data['subj'][key]))
                # print('subject type:    {}'.format(data['subj_type'][key]))
                # print("object:     {}".format(data['obj'][key]))
                # print("object type:     {}".format(data['obj_type'][key]))
                # print('real relation:   {}'.format(data['rel'][key]))
                # for i in range(len(predictions_l[key])):
                #     p_rel=predictions_l[key][i]
                #     print('1.Question relation:   {}'.format(predictions_l[key][i]))
                #     print('2.Question:    {}'.format(questions_l[key][i]))
                #     if "<obj>" in questions_l[key][i]:
                #         print('3.subj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['subj']))
                #         print('\n')
                #     if "<subj>" in questions_l[key][i]:
                #         print('3.obj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['obj']))
                #         print('\n')
                save_false['text'].append(data['text'][key])
                save_false['subject'].append(data['subj'][key])
                save_false['subject_type'].append(data['subj_type'][key])
                save_false['object'].append(data['obj'][key])
                save_false['object_type'].append(data['obj_type'][key])
                save_false['real relation'].append(data['rel'][key])
                tmp_res1=[]
                tmp_res2=[]
                tmp_res3=[]
                for i in range(len(predictions_l[key])):
                    p_rel = predictions_l[key][i]
                    # print('1.Question relation:   {}'.format(predictions_l[key][i]))
                    tmp_res1.append(predictions_l[key][i])
                    # print('2.Question:    {}'.format(questions_l[key][i]))
                    tmp_res2.append(questions_l[key][i])
                    if "<obj>" in questions_l[key][i]:
                        # print('3.subj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['subj']))
                        tmp_res3.append(rel_type[p_rel]['subj'])
                        # print('\n')
                    if "<subj>" in questions_l[key][i]:
                        # print('3.obj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['obj']))
                        tmp_res3.append(rel_type[p_rel]['obj'])
                        # print('\n')
                save_false['Question relation'].append(tmp_res1)
                save_false['Question'].append(tmp_res2)
                save_false['obj type of question'].append(tmp_res3)

        # elif len(predictions[key]) > 1:
        #
        #     """
        #     处理conflicts
        #     """
        #     filter_ans=set()
        #
        #     num_conflits += 1
        #     p_rels=predictions_l[key]
        #     qs=questions_l[key]
        #     rel=data['rel'][key]
        #     # res=results[key]
        #     res_tp=res_type[key]
        #
        #     p_rel_1=[]
        #     qs_1=[]
        #     require_types=[]
        #     # require_types_1=[]
        #     for p_rel,q in zip(p_rels,qs):
        #         if "<obj>" in q:
        #             # require_types_1.append(rel_type[p_rel]['obj_type'])
        #             if res_tp in rel_type[p_rel]['subj']:# or 'UNKNOWN' in rel_type[p_rel]['subj']:
        #                 filter_ans.update([p_rel])
        #                 p_rel_1.append(p_rel)
        #                 qs_1.append(q)
        #                 require_types.append(rel_type[p_rel]['subj'])
        #             # qs_1.append(q)
        #             # require_types.append(objtypeof_rel[p_rel])
        #         if "<subj>" in q:
        #             # require_types_1.append(rel_type[p_rel]['subj_type'])
        #             if res_tp in rel_type[p_rel]['obj']:# or 'UNKNOWN' in rel_type[p_rel]['obj']:
        #                 filter_ans.update([p_rel])
        #                 p_rel_1.append(p_rel)
        #                 qs_1.append(q)
        #                 require_types.append(rel_type[p_rel]['obj'])
        #             # qs_1.append(q)
        #             # require_types.append(objtypeof_rel[p_rel])
        #     if len(filter_ans) ==1:
        #         # save_result(key,p_rel_1[0],data)
        #         num_con_solved+=1
        #         if list(filter_ans)[0]==rel:
        #             num_solved_true+=1
        #             label_diversity.update(rel)
        #         # else:
        #             #solved_false
        #             # print('*' * 20)
        #             # print('text:    {}'.format(data['text'][key]))
        #             # print("subject:    {}".format(data['subj'][key]))
        #             # print('subject type:    {}'.format(data['subj_type'][key]))
        #             # print("object:     {}".format(data['obj'][key]))
        #             # print("object type:     {}".format(data['obj_type'][key]))
        #             # print('real relation:   {}'.format(data['rel'][key]))
        #             # # print(len(predictions_l[key]))
        #             # for i in range(len(predictions_l[key])):
        #             # 	q_tmp = str(questions_l[key][i]).replace('<subj>', data['subj'][key])
        #             # 	q_tmp = q_tmp.replace('<obj>', data['obj'][key])
        #             #
        #             # 	p_rel = predictions_l[key][i]
        #             # 	q=questions_l[key][i]
        #             # 	if "<obj>" in q and res_tp in rel_type[p_rel]['subj']:
        #             # 		print('1.Question relation:   {}'.format(predictions_l[key][i]))
        #             # 		print('2.Question:    {}'.format(q_tmp))
        #             # 		print(
        #             # 			'3.subj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['subj']))
        #             # 		print('\n')
        #             # 	if "<subj>" in q and res_tp in rel_type[p_rel]['obj']:
        #             # 		print('1.Question relation:   {}'.format(predictions_l[key][i]))
        #             # 		print('2.Question:    {}'.format(q_tmp))
        #             # 		print('3.obj types corresponding to the Questions:  {}'.format(rel_type[p_rel]['obj']))
        #             # 		print('\n')
        #             # num_con_0+=1
        #     # print('1.Question relation:   {}'.format(p_rels))
        #     # print('2.Question:    {}'.format(qs))
        #     # print('3.object types corresponding to the Quetions:  {}'.format(require_types_1))
        #     # print('4.real relation:   {}'.format(data['rel'][key]))
        #     # print('5.real object:     {}'.format(results[key]))
        #     # print('6.real object type:    {}'.format(res_type[key]))
        #     # print('\n')
        #     elif len(filter_ans) >1:
        #         # todo conflict_no_solved
        #         # cnt_no_conflicts+=1
        #         # cnt+=1
        #         num_con_nsolved+=1
        #         # print('0.text   :{}'.format(data['text'][key]))
        #         # print('1.Question relation:   {}'.format(p_rel_1))
        #         # print('2.Question:    {}'.format(qs_1))
        #         # print('3.types corresponding to the Quetions:  {}'.format(require_types))
        #         # print('4.real relation:   {}'.format(data['rel'][key]))
        #         # print('5.real ans:     {}'.format(results[key]))
        #         # print('6.real ans type:    {}'.format(res_type[key]))
        #         # print('7.subj:  {}'.format(data['subj'][key]))
        #         # print('8.obj:   {}'.format(data['obj'][key]))
        #         # print('\n')
        #         # for i,j in zip(p_rel_1,qs_1):
        #         #     if i!=data['rel'][key]:
        #         #         my_count.update([(data['rel'][key],i)])
        # else:
        #     #类型不对应 既不conflict也不no_conflict
        #     pass

        # print(predictions[key])
        # print(questions[key])
        # print(data['rel'][key])
        # print(results[key])
        # print(res_type[key])



        # print(data['subj_type'][key])
        # print(data['obj_type'][key])

    # print(label_diversity)
    # print('relation with corresponding object type'+'*'*50)
    # for i,j in rel2type.items():
    #     print(i,end="   : ")
    #     print(j)
    # print('*'*50)
    #     my_count=sorted(my_count.items(), key=lambda k: -k[1])
    #     for i in my_count:
    #         if i[1]>100:
    #             print(i[0])
    #             print(i[1])
    # print(my_count.most_common(10))
    # print(my_count_1.most_common(10))

# ----------------------------------------------------------------------------------------------------------------------


    """
    print 信息
    """
    print('num_no_conflits: {}'.format(num_no_conflits))
    print('num_correct:     {}'.format(num_correct))
    print()
    print('num_conflits:    {}'.format(num_conflits))
    print('answer_wrong: {}'.format(len(golds) - len(questions_l)))
    print()
    # print(num_con_nsolved)
    #
    print('num_con_solved:   {}'.format(num_con_solved))
    print('num_con_solved_true:     {}'.format(num_solved_true))
    # print('num_con_nsolved:     {}'.format(num_con_nsolved))

    print('-'*50)
    # for i,j in zip(my_count.keys(),my_count.values()):
    #     if j>40:
    #         print('{}----{}'.format(i,j))
    # print(sum(my_count.values()))
    # for i,j in zip(my_count.keys(),my_count.values()):
    # 	print(i)
    # 	print(j)
    # print(num_no_relation)
    print('-'*50)
    # print('answer wrong:    {}'.format(cnt_answer_wrong))
    # print('num_no_relation and answer_wrong: {}'.format(len(golds) - len(questions_l)))
    # print('conflicts_no_solved:    {}'.format(cnt))


    # print('the whole class number:      {}'.format(len(rel_type.keys())))
    # print('true label class number:     {}'.format(len(label_diversity.keys())))
    for i in rel_type.keys():
        if i not in label_diversity.keys():
            print(i)

    with open('data/output/save_ans.pkl',mode='wb') as fout1:
        pickle.dump(save_res,fout1)
    # with open('save_false.pkl',mode='wb') as fout1:
    #     pickle.dump(save_false,fout1)

    # print(num_no_relation)
    # print(len(golds))
    # print(len(predictions.keys()))
    # print(num_conflits)
    # print(num_no_conflits)


    # print(sum(my_count.values()))
    # print('conflicts_no_solved:    {}'.format(cnt_no_conflicts))
    # print('num_con_0:    {}'.format(num_con_0))

    # with open(join('data/output/','pretrain_data_3b_wiki.pkl'),mode='wb') as f:
    #     pickle.dump(save_res,f)
    # print(len(save_res['text']))
    # for i in save_res['text']:
    # 	print(i)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();

    # parser.add_argument('--dirname',type=str,default='qa_output_small_train_t5-base')
    parser.add_argument('--file_name', type=str, default='small_train',
                        help='name of input data')
    parser.add_argument('--model_name', type=str, default='t5-base',
                        help='name of pretrain T5 model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--data_dir', type=str, default='/home/zzengae/inferT5/data/tacred',
                        help='data directory')
    parser.add_argument('--output_identifier', type=str, default='qa',
                        help='data directory')
    parser.add_argument('--out_data_dir', type=str, default='./data',
                        help='output for process')
    parser.add_argument('--input_data_dir', type=str, default='data/',
                        help='input=>data.pkl')


    args=parser.parse_args()

    main(args)

    """
    next=>处理new_save_ans.pkl(crosses and religion)=>read_pkl.py
    """
