python qnli_T5.py \
--file_name wiki80_train \
--model_name t5-3b \
--batch_size 128 \
--data_dir data/data_aid \
--sents_batch_size 1000 \
--output_data_dir data/output \
--input_data_dir data/type_data/type_infer_output_standford_wiki80_train_t5-base/data_final.pkl \
--flag INCVST
--use_gpucd

#echo qqqqqqqqqqqqqqqqqqqq
#
#python prepare_qa_input.py --file_name wiki80_train \
#--model_name t5-3b \
#--data_dir data/data_aid \
#--sents_batch_size 1000 \
#--input_data_path data/data/type_infer_output_standford_wiki80_train_t5-base/data.pkl \
#--output_data_dir data/output/

#echo qqqqqqqqqqqqqqqqqqqq
#

#python qa_T5.py --file_name wiki80_train \
#--model_name t5-3b \
#--batch_size 128 \
#--data_dir data/data_aid \
#--out_data_dir data/output \
#--verbose \
#--use_gpu

#
#python conflict_items.py --data_dir data/data_aid \
#--out_data_dir ./data/output \
#--file_name wiki80_train \
#--model_name t5-3b \
#--output_identifier qa \
#--input_data_dir data/data/type_infer_output_standford_wiki80_train_t5-base/data.pkl

#python type_infer.py


#--------------------------------------------------------------------------------------------------
#
#python process_qa_output.py --file_name test \
#--model_name 't5-3b' \
#--output_data_dir data/output \
#--data_dir data
##
#python prepare_qg_input.py --file_name train \
#--model_name t5-base \
#--out_data_dir data/output \
#--data_dir /data \

#python qg_T5.py --file_name small_train \
#--model_name t5-base \
#--batch_size 80 \
#--use_gpu \
#--verbose

#!no_use
#python fast_qnli_T5.py --file_name small_train \
#--model_name t5-base \
#--data_dir /home/zzengae/inferT5/data/tacred \
#--sents_batch_size 1000

#python process_qg_output.py --bound 0


