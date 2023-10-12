import os
import argparse
import numpy as np

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='HyperHawkes')
    parser.add_argument('--dataset', '-d', type=str, default='rees46')

    parser.add_argument('--use-hgnn', '-hgnn', type=int, default=1)
    parser.add_argument('--use-atten-mixer', '-am', type=int, default=1)
    parser.add_argument('--use-base-excitation', '-be', type=int, default=1)
    parser.add_argument('--use-self-item-excitation', '-site', type=int, default=1)
    parser.add_argument('--use-self-intent-excitation', '-sine', type=int, default=1)

    args, _ = parser.parse_known_args()
    config_file_list = [f"./configs/general_full.yaml", f"./configs/dataset/{args.dataset}.yaml"]

    model_config_path = f"./configs/model/{args.model}/{args.dataset}.yaml"
    if os.path.exists(model_config_path):
        config_file_list.append(model_config_path)

    run_recbole_gnn(model=args.model, dataset=args.dataset,
                    config_file_list=config_file_list,
                    config_dict=vars(args))
    print(args.dataset)