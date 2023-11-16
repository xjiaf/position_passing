import random
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logging(save_path, log_level=logging.INFO):
    """
    Set up logging to file and console.
    """
    # Create the directory if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    filename = str(Path(save_path) / 'run.log')

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(filename, mode='a', delay=True)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)


def download_data(params: dict):
    """
    This function is used to process the data and generate the graph.
    """
    from torch_geometric.datasets import JODIEDataset

    # Set the path where data will be saved
    path = params['data_path']

    # Create a list containing the names of all datasets to be downloaded
    dataset_names = ["Reddit", "Wikipedia", "MOOC", "LastFM"]

    # Download each dataset by its name
    for name in dataset_names:
        _ = JODIEDataset(root=path, name=name)


def main(params):
    setup_seed(params['seed'])
    if params['mode'] == 'data':
        save_path = Path(params['data_path'])
    else:
        save_path = Path(
            params['result_path'], params['dataset'], params['model'])
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path)
    logging.info(save_path)
    if params['mode'] == 'data':
        pass
        # download_data(params)
    elif params['mode'] == 'train':
        from utils.train_test import Trainer
        trainer = Trainer(params, device=device)
        trainer.train()
        logging.info('Training finished, start testing...')
        trainer.test()
        model_save_path = save_path / 'model.pt'
        torch.save(trainer.model.state_dict(), model_save_path)

    elif params['mode'] == 'test':
        from utils.train_test import Trainer
        tester = Trainer(params, save_path=save_path, device=device)
        tester.test()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_params(args, config):
    params = {**config['default']}
    params['mode'] = str(args.mode)
    # If model argument is provided, merge model-specific params
    if params['mode'] != 'data':
        params = {**config['default'], 
                  **config['datasets'][args.dataset]}
        params['mode'] = str(args.mode)
        params = {**params, **config['models'][args.model]}

    # Ensure paths are cross-platform compatible
    params['data_path'] = Path(params['data_path'])
    params['processed_data_path'] = Path(params['processed_data_path'])
    params['result_path'] = Path(params['result_path'])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training and Testing TGSL.")
    parser.add_argument('--dataset', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm'],
                        help='Which dataset to use.')
    parser.add_argument('--mode', type=str, choices=[
        'train', 'test', 'data'], default='train')
    # parser.add_argument('--model', type=str, choices=[
    #     'dgnn'], help='Which model to use.')
    args = parser.parse_args()
    config = load_config('./config.yaml')
    params = get_params(args, config)
    try:
        print(params)
        main(params)
    except Exception:
        logging.exception("An error occurred!")
