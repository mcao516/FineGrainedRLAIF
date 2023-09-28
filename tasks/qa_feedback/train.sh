echo "========== SETTINGS =========="
set -e
DATA_FOLDER=toy
OUTPUT_FOLDER_NAME=test

echo $DATA_FOLDER
echo $OUTPUT_FOLDER_NAME

echo "========== DATE PREPROCESSING =========="
python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level subsentence --error_category NF-ERR --ignore_context --data_dir ./tasks/qa_feedback/${DATA_FOLDER}/
python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level sentence --error_category F-ERR --data_dir ./tasks/qa_feedback/${DATA_FOLDER}/
python tasks/qa_feedback/reward_modeling/create_comp_rm_files.py --input_dir ./tasks/qa_feedback/${DATA_FOLDER}/ --output_dir ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/


echo "========== REL REWARD MODELING =========="
REL_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/rel_rm

# train reward model for NF-ERR_subsentence
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/NF-ERR_subsentence/train.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/NF-ERR_subsentence/dev.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/NF-ERR_subsentence/dev.json \
                --output_dir $REL_OUTPUT_DIR \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 12 \
                --per_device_eval_batch_size 12 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.01 \
                --warmup_ratio 0.1;


echo "========== FACT REWARD MODEL =========="
FACT_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/fact_rm

# train reward model for F-ERR_sentence
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/F-ERR_sentence/train.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/F-ERR_sentence/dev.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/F-ERR_sentence/dev.json \
                --output_dir $FACT_OUTPUT_DIR \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 12 \
                --per_device_eval_batch_size 12 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1;


echo "========== COMP REWARD MODEL =========="
COMP_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/comp_rm

# train reward model for COMP
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --output_dir $COMP_OUTPUT_DIR \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 24 \
                --eval_steps 200 \
                --evaluation_strategy steps \
                --logging_steps 200 \
                --logging_strategy steps \
                --save_steps 200 \
                --save_strategy steps \
                --load_best_model_at_end \
                --metric_for_best_model accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.00005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1 \
                --remove_unused_columns False;


# inference for getting mean std of COMP
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path $OUTPUT_DIR \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --output_dir $COMP_OUTPUT_DIR \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 64 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True;


echo "========== PPO =========="
accelerate launch \
    --main_process_port 29500 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 4 \
    --mixed_precision bf16 \
    --multi_gpu \
    tasks/qa_feedback/training/train_finegrained.py \
        --config tasks/qa_feedback/training/fine_grained_config.yml \
        --rm_rel_ckpt $REL_OUTPUT_DIR \
        --rm_fact_ckpt $FACT_OUTPUT_DIR \
        --rm_comp_ckpt $COMP_OUTPUT_DIR \
        --mean_std_file_path $COMP_OUTPUT_DIR/mean_std.txt \
        --save_dir ./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/fine_grained/;
