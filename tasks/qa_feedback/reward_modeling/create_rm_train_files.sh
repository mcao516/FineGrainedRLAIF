set -e
DATA_FOLDER=data

python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level subsentence --error_category NF-ERR --ignore_context --data_dir ./tasks/qa_feedback/${DATA_FOLDER}/
python tasks/qa_feedback/reward_modeling/create_rel_fact_rm_files.py --feedback_level sentence --error_category F-ERR --data_dir ./tasks/qa_feedback/${DATA_FOLDER}/
python tasks/qa_feedback/reward_modeling/create_comp_rm_files.py --input_dir ./tasks/qa_feedback/${DATA_FOLDER}/ --output_dir ./tasks/qa_feedback/${DATA_FOLDER}/COMP_sequence/
