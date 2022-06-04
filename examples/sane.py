from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.solver.utils import set_seed
import argparse
from autogl.backend import DependentBackend
from torch_geometric.utils import add_self_loops
import torch
from sklearn.model_selection import StratifiedKFold
from torch import cat
from autogl.datasets.utils import random_splits_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/nodeclf_sane_benchmark.yml')
    parser.add_argument('--dataset', default='cora', type=str)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    set_seed(args.seed)

    dataset = build_dataset_from_name(args.dataset, path='/data/AutoGL')
    random_splits_mask(dataset, 0.6, 0.2)

    solver = AutoNodeClassifier.from_config(args.config)
    solver.fit(dataset)
    solver.get_leaderboard().show()
    acc = solver.evaluate(metric="acc")
    print('acc on dataset', acc)
