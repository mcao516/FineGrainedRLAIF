DATA_FOLDER=llm340bmix_data
OUTPUT_FOLDER_NAME=llm340bmix_data

echo $DATA_FOLDER
echo $OUTPUT_FOLDER_NAME

REL_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/rel_rm
FACT_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/fact_rm
COMP_OUTPUT_DIR=./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/comp_rm

accelerate launch \
    --main_process_port 29500 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 4 \
    --mixed_precision bf16 \
    --multi_gpu \
    tasks/qa_feedback/training/eval_finegrained.py \
        --config tasks/qa_feedback/training/fine_grained_config.yml \
        --rm_rel_ckpt $REL_OUTPUT_DIR \
        --rm_fact_ckpt $FACT_OUTPUT_DIR \
        --rm_comp_ckpt $COMP_OUTPUT_DIR \
        --mean_std_file_path $COMP_OUTPUT_DIR/mean_std.txt \
        --save_dir ./tasks/qa_feedback/model_outputs/${OUTPUT_FOLDER_NAME}/fine_grained/ \
        --folder_name $DATA_FOLDER;
