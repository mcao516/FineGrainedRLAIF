import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb
import yaml
import nltk

from fgrlhf.ppo import PPOTrainer
from fgrlhf.policy import T5Policy
from fgrlhf.value import T5Value
from fgrlhf.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

from reward import FineGrainedReward

logging.basicConfig(level=logging.ERROR)

# prepare accelerator and logger
accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="path to config file")
parser.add_argument("--policy_ckpt", required=True, type=str, help="path to policy model checkpoint")
parser.add_argument("--rm_rel_ckpt", required=True, type=str, help="path to relevance reward model checkpoint")
parser.add_argument("--rm_fact_ckpt", required=True, type=str, help="path to factuality reward model checkpoint")
parser.add_argument("--rm_comp_ckpt", required=True, type=str, help="path to completeness reward model checkpoint")
parser.add_argument("--mean_std_file_path", required=True, type=str, help="path to mean and std file")
parser.add_argument("--save_dir", required=True, type=str, help="path to save model")
parser.add_argument("--folder_name", default="data", type=str)
args = parser.parse_args()
# load yaml file
with open(args.config) as f:
    policy_ckpt = args.policy_ckpt
    rm_rel_ckpt = args.rm_rel_ckpt
    rm_fact_ckpt = args.rm_fact_ckpt
    rm_comp_ckpt = args.rm_comp_ckpt
    save_dir = args.save_dir
    mean_std_file_path = args.mean_std_file_path
    folder_name = args.folder_name
    args =yaml.safe_load(f)

    # overwrite ckpt path
    args['policy_model_to_eval'] = policy_ckpt
    args['reward']['relevance_model']['ckpt'] = rm_rel_ckpt
    args['reward']['factuality_model']['ckpt'] = rm_fact_ckpt
    args['reward']['completeness_model']['ckpt'] = rm_comp_ckpt
    args['logging']['save_dir'] = save_dir
    args['mean_std_file_path'] = mean_std_file_path
    args['folder_name'] = folder_name
    args['logging']['wandb_project'] = "FineGrainedRLHF-Eval"
    args['logging']['run_name'] = "evaluation"


# prepare data
class TextGenDataset(Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        accelerator=None,
        length_limit=None,
        folder_name="data"
    ):
        super().__init__()
        
        self.split = split
        self.dataset_fns = {
            "train": f"tasks/qa_feedback/{folder_name}/train.json",
            "dev": f"tasks/qa_feedback/{folder_name}/dev.json",
            "test": f"tasks/qa_feedback/{folder_name}/test.json"
        }
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:
            instances.append({
                "prompt": task_instance['text'],
                "metadata": {
                    "prompt": task_instance['text'],
                    "references": task_instance['answer'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],}
            })
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata
        }
        return result
    

def read_mean_std(file_path):
    with open(file_path, 'r') as f:
        mean = float(f.readline().strip())
        std = float(f.readline().strip())
    return mean, std


def valid(args, accelerator, eval_dataloader, policy_model, reward_model):
    accelerator.wait_for_everyone()
    policy_model.model.eval()

    columns=["step", "inputs", "outputs"]
    wandb_table = None
    
    n_entries = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader) if accelerator.is_main_process else eval_dataloader):

            results = policy_model.sample(
                prompts_input_ids=batch['prompts_input_ids'],
                prompts_attention_mask=batch['prompts_attention_mask'],
                **args['model']['policy_model']['eval_generation_kwargs'],
            )

            eval_results = reward_model.eval_metrics(
                prompts_input_ids=results['prompts_input_ids'],
                prompts_attention_mask=results['prompts_attention_mask'],
                generated_input_ids=results['generated_input_ids'],
                generated_attention_mask=results['generated_attention_mask'],
                generated_texts=results['generated_text'],
                metadata = batch['metadata'],
            )
            
            # gather all results
            batch = accelerator.gather_for_metrics(batch)
            results = accelerator.gather_for_metrics(results)
            
            for eval_k, eval_v in eval_results.items():
                eval_results[eval_k] = accelerator.gather(
                    torch.tensor(eval_v, device=results['generated_input_ids'].device))
                
            # initialize wandb table if it does not exist
            if wandb_table is None:
                columns.extend(list(eval_results.keys())) 
                wandb_table = wandb.Table(columns=columns)
            
            if accelerator.is_main_process: 
                prompt_inputs = policy_model.tokenizer.batch_decode(results['prompts_input_ids'],
                                                                skip_special_tokens=True, 
                                                                clean_up_tokenization_spaces=True)
                generated_texts = policy_model.tokenizer.batch_decode(results['generated_input_ids'],
                                                                skip_special_tokens=True, 
                                                                clean_up_tokenization_spaces=True)
                
                this_data_batch_size = results['prompts_input_ids'].shape[0]
                this_lens = torch.sum(results['generated_attention_mask'], dim=-1)
                
                for batch_i in range(this_data_batch_size):
                    this_entry = [0, prompt_inputs[batch_i], generated_texts[batch_i]]
                    
                    for eval_v in eval_results.values():
                        this_entry.append(eval_v[batch_i].item())
                    
                    wandb_table.add_data(*this_entry)
                    n_entries += 1

    if accelerator.is_main_process:        
        # stats = {'eval/step': 0, f'eval_generation/step_{0}': wandb_table}
        stats = {'eval/step': 0}

        value_columns = columns[3:] # the first three are steps, inputs, outputs
        stats.update(reward_model.aggregate_metrics(wandb_table, value_columns))

        if args['logging']['wandb_log']:
            wandb.log(stats)

        for k, v in stats.items():
            log_info(f'Evaluated [{k}] = {v:.4f}')

        print("##### Table #####")
        print(wandb_table)


def main():
    # set seed
    set_seed(args['train']['seed'], args['train']['cuda_deterministic'])

    # set saving directories
    log_info(f"Write to output directory: {args['logging']['save_dir']}")

    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        # save the config file
        with open(os.path.join(args['logging']['save_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    # initialize policy and value model tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                           model_max_length=args['env']['max_input_len'])
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']

    # Load data
    log_info(f'Loading evaluation data ...')
    eval_dataset = TextGenDataset('dev',  tokenizer, accelerator=accelerator, length_limit=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16,
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    if os.path.isfile(args['policy_model_to_eval']) and \
        not os.path.isdir(args['policy_model_to_eval']):
        log_info(f"Load policy model from: {args['policy_model_to_eval']}")
        policy = T5Policy(
            model_ckpt=args['model']['policy_model']['ckpt'],
            tokenizer=tokenizer,
            policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
            accelerator=accelerator,
        )
        policy.model.load_state_dict(torch.load(args['policy_model_to_eval']))
    else:
        log_info(f"Load policy model from: {args['policy_model_to_eval']}")
        policy = T5Policy(
            model_ckpt=args['policy_model_to_eval'],
            tokenizer=tokenizer,
            policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
            accelerator=accelerator,
        )
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)

    reward = FineGrainedReward(
        tokenizer=tokenizer,
        non_factual_model_ckpt=args['reward']['relevance_model']['ckpt'],
        factual_model_ckpt=args['reward']['factuality_model']['ckpt'],
        completeness_model_ckpt=args['reward']['completeness_model']['ckpt'],
        kl_coef=args['ppo']['kl_coef'],
        verbosity_positive_reward=args['reward']['relevance_model']['positive_reward'],
        verbosity_negative_reward=args['reward']['relevance_model']['negative_reward'],
        factuality_positive_reward=args['reward']['factuality_model']['positive_reward'],
        factuality_negative_reward=args['reward']['factuality_model']['negative_reward'],
        completeness_reward_mean=args['reward']['completeness_model']['mean'],
        completeness_reward_std=args['reward']['completeness_model']['std'],
        completeness_reward_bias=args['reward']['completeness_model']['bias'],
        completeness_reward_scale=args['reward']['completeness_model']['scale'],
        sep="</s>"
    )
    
    # prepare reward models
    reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
    reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
    reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)

    # Set up trainer
    # trainer = PPOTrainer(
    #     args=args,
    #     train_dataloader=train_dataloader,
    #     eval_dataloader=eval_dataloader,
    #     ref_policy_model=None,
    #     policy_model=policy,
    #     value_model=None,
    #     reward_model=reward,
    #     optimizer=None,
    #     scheduler=None,
    #     accelerator=accelerator,
    #     log_info=log_info,
    # )

    if accelerator.is_main_process:
        if args['logging']['wandb_log']:
            wandb.init(entity=args["logging"]["wandb_entity"], project=args["logging"]["wandb_project"], name=args['logging']['run_name'], config=args)
        else:
            wandb.init(config=args, mode='disabled')

        wandb.define_metric('train/step')
        wandb.define_metric('eval/step')
        wandb.define_metric('train/*', step_metric='train/step')
        wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

    valid(args, accelerator, eval_dataloader, policy, reward)

            
if __name__ == '__main__':
    main()
