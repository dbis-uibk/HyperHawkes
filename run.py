import os
import argparse
import time

from recbole_custom.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='HyperHawkes', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ta-feng', help='name of datasets')
    parser.add_argument('--show-progress', '-sp', type=int, default=0)
    parser.add_argument('--seed', '-s', type=int, default=2023)
    
    args, _ = parser.parse_known_args()

    config_file_list = [f"./configs/general_full.yaml"]

    config_file_list.append(f"./configs/dataset/{args.dataset}.yaml")
    model_config_path = f"./configs/model/{args.model}/{args.dataset}.yaml"
    if os.path.exists(model_config_path):
        config_file_list.append(model_config_path)

    print(config_file_list)

    if args.seed is None or args.seed == 0:
        args.seed = int(time.time() // 1000)

    run_recbole(model=args.model, dataset=args.dataset,
                    config_file_list=config_file_list,
                    config_dict=vars(args))
    print(args.dataset)
    print(args)