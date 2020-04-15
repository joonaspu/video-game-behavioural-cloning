#!/usr/bin/env python3
#
# get_vizdoom_scores.py
# Simply print out the scores of human ViZDoom recordings
#

import argparse
import json
from pprint import pprint

import numpy as np


parser = argparse.ArgumentParser("Print out scores from recorded ViZDoom games")
parser.add_argument("inputs", type=str, nargs="+", help="Input .json files")


def main(args):
    scores = []
    for filepath in args.inputs:
        json_data = None
        with open(filepath) as f:
            json_data = json.load(f)
        rewards = [step["r"] for step in json_data["steps"]]
        scores.append(sum(rewards))

    print("Individual scores: ")
    pprint(scores)

    print("Mean: {:.3f}. Std: {:.3f}".format(
        np.mean(scores),
        np.std(scores)
    ))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
