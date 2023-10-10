import argparse
import transformers
from reward import SubSentenceVerbosityReward, FactualityReward, PreferenceReward, FineGrainedReward


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.policy_model_ckpt, 
                                                           model_max_length=1024)

    reward_model = FineGrainedReward(
        tokenizer=tokenizer,
        non_factual_model_ckpt=args.relevance_model_ckpt,
        factual_model_ckpt=args.factuality_model_ckpt,
        completeness_model_ckpt=args.completeness_model_ckpt,
        kl_coef=0.3,
        verbosity_positive_reward=0.3,
        verbosity_negative_reward=-0.3,
        factuality_positive_reward=0.5,
        factuality_negative_reward=-0.5,
        completeness_reward_mean=-0.44677690555995353,
        completeness_reward_std=8.301160619054132,
        completeness_reward_bias=0.0,
        completeness_reward_scale=0.3,
        sep="</s>"
    )

    reward_results = reward_model.get_reward(
        prompts_input_ids=results['prompts_input_ids'],
        prompts_attention_mask=results['prompts_attention_mask'],
        generated_input_ids=results['generated_input_ids'],
        generated_attention_mask=results['generated_attention_mask'],
        generated_texts=results['generated_text'],
        metadata = [elem for elem in batch['metadata'] for _ in range(self.args['env']['train_num_samples_per_input'])],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_model_ckpt", default="lvwerra/gpt2-imdb", type=str)
    parser.add_argument("--relevance_model_ckpt", default="lvwerra/gpt2-imdb", type=str)
    parser.add_argument("--factuality_model_ckpt", default="lvwerra/gpt2-imdb", type=str)
    parser.add_argument("--completeness_model_ckpt", default="lvwerra/gpt2-imdb", type=str)

    args = parser.parse_args()
    main(args)