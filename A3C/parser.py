import argparse
import torch
import multiprocessing as mp

parser = argparse.ArgumentParser(description="A3C")
parser.add_argument(
    "-l", "--lr", type=float, default=1e-5, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "-num_processes", type=int, default=15, help="Number of processes"
)
parser.add_argument(
    "-epochs", type=int, default=30000, help="Number of epochs"
)
parser.add_argument(
    "-g","--gamma",type=float,default=0.99,help="discount factor for rewards (default: 0.98)"
)
parser.add_argument(
    "-hidden_size",type=int,default=128,help="number of hidden layer's nue"
)
parser.add_argument(
    "-input_size",type=int,default=24,help="size of input tensor"
)
parser.add_argument(
    "-clc",type=float,default=0.5
)
parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient for exploration')
