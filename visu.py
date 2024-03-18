import matplotlib.pyplot as plt
from utils.utils import load_learn_hists, visualize_learn_hists
import os
import argparse

def generate_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize the learning histories of different experiments')
    parser.add_argument('exp_names', type=str, nargs='+', help='Paths to the learning histories files')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot')
    parser.add_argument('--title', type=str, default='Learning histories', help='Title of the plot')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory where the results are saved.')
    return parser

def find_json_file(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                return os.path.join(root, file)
    return None

def main(args):
    results_dir = args.results_dir
    experiments_paths = [find_json_file(os.path.join(results_dir, exp_name)) for exp_name in args.exp_names]
    
    
    # Parse the experiments_path folder and find the path to the json file.
    learn_hists = [load_learn_hists(exp_json_path) for exp_json_path in experiments_paths]
    print(learn_hists)
    visualize_learn_hists(learn_hists, args.save_path)
    

if __name__ == '__main__':
    parser = generate_argument_parser()
    args = parser.parse_args()
    main(args)