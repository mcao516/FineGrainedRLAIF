echo "===================== COMP =========================="
DATA_FOLDER=llm_train_human_dev
OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${DATA_FOLDER}/comp_rm

# train reward model for COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --output_dir $OUTPUT_DIR \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 6 \
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
                --remove_unused_columns False


# inference for getting mean std of COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path $OUTPUT_DIR \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --output_dir $OUTPUT_DIR \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 32 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True

DATA_FOLDER=llm_data
OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${DATA_FOLDER}/comp_rm

# train reward model for COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/dev.json \
                --output_dir $OUTPUT_DIR \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 6 \
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
                --remove_unused_columns False


# inference for getting mean std of COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path $OUTPUT_DIR \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/train.json \
                --output_dir $OUTPUT_DIR \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 32 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True

echo "==================================== BASELINE ========================================"
DATA_FOLDER=data
OUTPUT_DIR=./tasks/qa_feedback/model_outputs/baseline_rm

# train reward model for baseline
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/${DATA_FOLDER}/train_feedback.json \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/dev_feedback.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/dev_feedback.json \
                --output_dir $OUTPUT_DIR \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 6 \
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
                --learning_rate 0.00001 \
                --weight_decay 0.01 \
                --warmup_ratio 0.1 \
                --remove_unused_columns False


# inference for getting mean std of baseline
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path $OUTPUT_DIR \
                --validation_file ./tasks/qa_feedback/${DATA_FOLDER}/train_feedback.json \
                --test_file ./tasks/qa_feedback/${DATA_FOLDER}/train_feedback.json \
                --output_dir $OUTPUT_DIR \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 32 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True