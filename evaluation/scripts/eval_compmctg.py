import subprocess
import argparse

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--eval_Fyelp_path", default="./scripts/eval_acc_Fyelp.py", type=str)
    parser.add_argument("--eval_Amazon_path", default="./scripts/eval_acc_Amazon.py", type=str)
    parser.add_argument("--eval_Yelp_path", default="./scripts/eval_acc_Yelp.py", type=str)
    parser.add_argument("--eval_Mixture_path", default="./scripts/eval_acc_Mixture.py", type=str)
    parser.add_argument("--eval_perplexity_path", default="./scripts/eval_perplexity.py", type=str)
    parser.add_argument("--dataset", default=None, type=str, choices=["Fyelp", "Amazon", "Yelp", "Mixture"])
    parser.add_argument("--device_num", default=None, type=str)
    args = parser.parse_args()

    assert args.dataset is not None
    assert args.device_num is not None
    args_to_pass = ["--dataset_path", args.dataset_path, "--device_num", args.device_num]
    if args.dataset in args.eval_Fyelp_path:
        subprocess.run(['python', args.eval_Fyelp_path] + args_to_pass)
        subprocess.run(['python', args.eval_perplexity_path] + args_to_pass)
    elif args.dataset in args.eval_Amazon_path:
        subprocess.run(['python', args.eval_Amazon_path] + args_to_pass)
        subprocess.run(['python', args.eval_perplexity_path] + args_to_pass)
    elif args.dataset in args.eval_Yelp_path:
        subprocess.run(['python', args.eval_Yelp_path] + args_to_pass)
        subprocess.run(['python', args.eval_perplexity_path] + args_to_pass)
    elif args.dataset in args.eval_Mixture_path:
        subprocess.run(['python', args.eval_Mixture_path] + args_to_pass)
        subprocess.run(['python', args.eval_perplexity_path] + args_to_pass)
    else:
        raise Exception("Wrong dataset")

if __name__ == "__main__":
    eval()