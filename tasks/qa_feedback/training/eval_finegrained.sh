DATA_FOLDER=data
OUTPUT_FOLDER_NAME=data

echo $DATA_FOLDER
echo $OUTPUT_FOLDER_NAME

POLICY_OUTPUT_DIR=$HOME/FineGrainedRLAIF/tasks/qa_feedback/model_outputs/data2/fine_grained/best_policy.pth
POLICY_OUTPUT_DIR=$HOME/FineGrainedRLHF-Original/tasks/qa_feedback/model_outputs/llm_data/fine_grained/best_policy.pth

REL_OUTPUT_DIR=$HOME/FineGrainedRLAIF/fgrlhf_models/rel_rm
FACT_OUTPUT_DIR=$HOME/FineGrainedRLAIF/fgrlhf_models/fact_rm
COMP_OUTPUT_DIR=$HOME/FineGrainedRLAIF/fgrlhf_models/comp_rm

accelerate launch \
    --main_process_port 8899 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 4 \
    --mixed_precision bf16 \
    --multi_gpu \
        tasks/qa_feedback/training/eval_finegrained.py \
            --config tasks/qa_feedback/training/fine_grained_config.yml \
            --policy_ckpt $POLICY_OUTPUT_DIR \
            --rm_rel_ckpt $REL_OUTPUT_DIR \
            --rm_fact_ckpt $FACT_OUTPUT_DIR \
            --rm_comp_ckpt $COMP_OUTPUT_DIR \
            --mean_std_file_path $COMP_OUTPUT_DIR/mean_std.txt \
            --save_dir ./eval \
            --folder_name $DATA_FOLDER;
