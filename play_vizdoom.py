import os
from argparse import ArgumentParser
from multiprocessing import Process, Queue, set_start_method
from random import randint
from functools import reduce

parser = ArgumentParser("Play and evaluate vizdoom games with a trained network.")
parser.add_argument("models", type=str, nargs="+",
                    help="Path of the file(s) where the model will be loaded from.")
parser.add_argument("--save", "-s", type=str, nargs="?", default="./results",
                    help="Path where the results of the evaluation will be saved.")
parser.add_argument("--config", type=str, required=True,
                    help="Path to the vizdoom config file")
parser.add_argument("--framestack", type=int, default=3,
                    help="Number of frames to stack (must match the number used in model)")
parser.add_argument("--rate", type=int, default=2,
                    help="Aka frameskip, number of frames per prediction")
parser.add_argument("--width", "-x", type=int, default=84,
                    help="Width of the image")
parser.add_argument("--height", "-y", type=int, default=84,
                    help="Height of the image")
parser.add_argument("--display", action="store_true",
                    help="Display gameplay in a window")
parser.add_argument("--processes", type=int, default=1,
                    help="How many parallel processes to run.")
parser.add_argument("--games", type=int, default=1,
                    help="How many games (per process) to run.")
parser.add_argument("--no-cuda", action="store_true",
                    help="Disable CUDA")
parser.add_argument("--random", action="store_true",
                    help="Ignore model and just pick random actions.")

args = parser.parse_args()

import numpy as np
import gym
from PIL import Image, ImageChops
import vizdoom as vzd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import Mnih2015

if args.no_cuda:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_game(model_name, queue, index):
    """Plays one game with the given model and gym environment
       and returns the final score (i.e. cumulative reward)"""

    print("Starting process #{}..".format(index))

    if not args.random:
        model = torch.load(model_name, map_location=device)
        model.eval()

    env = vzd.DoomGame()

    env.load_config(args.config)

    if args.display:
        env.set_window_visible(True)
        # Set speed to a comfortable, human-enjoy
        # lelvel
        env.set_mode(vzd.Mode.ASYNC_PLAYER)
    else:
        env.set_mode(vzd.Mode.PLAYER)

    env.init()

    rng = np.random.default_rng()

    for game in range(args.games):
        env.new_episode()

        o = env.get_state()
        r, d, i = (0.0, False, None)

        total_reward = 0

        # Create a frame stack and fill it with zeros (black images)
        stack = []
        for _ in range(args.framestack):
            stack.append(np.zeros((args.width, args.height, 3), dtype=np.uint8))

        while True:
            # Resize image
            img = o.screen_buffer
            # ViZDoom gives images as 
            # CHW, turn to HWC
            img = img.transpose([1, 2, 0])
            img = Image.fromarray(img)
            img = img.resize((args.width, args.height), Image.BILINEAR)
            # Turn to float (normalization happens later)
            img = np.asarray(img, dtype=np.float32)

            # Update the frame stack
            stack.insert(0, img)
            while len(stack) > args.framestack:
                stack.pop()

            # Make sure we have enough frames stacked
            if len(stack) != args.framestack:
                continue

            # Convert stack to numpy array with correct dimensions and type
            np_stack = np.concatenate(stack, axis=2)
            np_stack = np.expand_dims(np_stack, axis=0)
            np_stack = np_stack.astype(np.float32)

            # Normalize
            np_stack /= 255

            if args.random:
                actions_num = env.get_available_buttons_size()
                prediction = np.random.randint(2, size=actions_num).tolist()
            else:
                # Get prediction
                prediction = model(torch.Tensor(np.swapaxes(np_stack, 1, 3)).to(device)).detach().cpu()[0]
                prediction = torch.sigmoid(prediction).numpy()

                # Convert prediction to a list of {0, 1} values for each control
                prediction = (np.random.random(size=prediction.shape) < prediction).astype(np.int)
                prediction = prediction.tolist()

            r = env.make_action(prediction, args.rate)
            d = env.is_episode_finished()

            total_reward += r
            if d:
                queue.put(total_reward)
                break
            else:
                # We can not ask for state if episode is finished
                o = env.get_state()

        print("#{} finished game {}".format(index, game))

def main():
    set_start_method("spawn")

    for model in args.models:
        # Get model name from path
        model_name = os.path.basename(os.path.normpath(model))

        # Make sure results directory exists
        results_path = os.path.normpath(args.save)
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        # Path to the results file
        results_name = "{}.txt".format(model_name)
        results_file = os.path.normpath(os.path.join(results_path, results_name))

        print("Evaluating model {}".format(model))

        rewards = Queue()
        procs = []

        # Start processes
        # Using threads doesn't work as the OpenAI Atari gym crashes if run
        # from multiple threads at the same time. Processes work fine though.
        for i in range(args.processes):
            proc = Process(target=play_game, args=(model, rewards, i))
            proc.start()
            procs.append(proc)

        # Wait for processes to finish
        for proc in procs:
            proc.join()

        # Collect results from processes
        with open(results_file, "w") as f:
            rewards_list = []
            while not rewards.empty():
                r = rewards.get()
                rewards_list.append(r)
                f.write("{}\n".format(r))
                print(r)
            
            if len(rewards_list) < 1:
                avg = 0
            else:
                avg = sum(rewards_list)/len(rewards_list)
            f.write("Avg: {}".format(avg))
            print("Avg: {}".format(avg))

if __name__ == "__main__":
    main()
