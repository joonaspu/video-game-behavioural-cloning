import os
from argparse import ArgumentParser
from multiprocessing import Process, Queue, set_start_method
from queue import Empty
import multiprocessing
from random import randint
from functools import reduce
import statistics

parser = ArgumentParser("Play and evaluate Atari games with a trained network.")
parser.add_argument("models", type=str, nargs="+",
                    help="Path of the file(s) where the model will be loaded from.")
parser.add_argument("--save", "-s", type=str, nargs="?", default="./results",
                    help="Path where the results of the evaluation will be saved.")
parser.add_argument("--env", type=str, default="SpaceInvaders-v0",
                    help="Name of the Atari environment to use")
parser.add_argument("--framestack", type=int, default=3,
                    help="Number of frames to stack (must match the number used in model)")
parser.add_argument("--merge", action="store_true",
                    help="Merge stacked frames into one image.")
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
parser.add_argument("--action", type=str, default="sampling",
                    choices=["sampling", "argmax"],
                    help="Use random sampling or argmax to pick actions.")
parser.add_argument("--no-op", type=int, default=0,
                    help="Maximum number of no-op actions at the beginning of each game.")
parser.add_argument("--max-frames", type=int, default=40000,
                    help="Maximum number of frames to run the game for before ending evaluation.")                
parser.add_argument("--no-cuda", action="store_true",
                    help="Disable CUDA")
parser.add_argument("--random", action="store_true",
                    help="Ignore model and just pick random actions.")

args = parser.parse_args()

import numpy as np
from PIL import Image, ImageChops
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import Mnih2015

if args.no_cuda:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_game(model_name, env_name, game_queue, reward_queue, index):
    """Plays one game with the given model and gym environment
       and returns the final score (i.e. cumulative reward)"""

    print("Starting process #{}..".format(index))

    if not args.random:
        model = torch.load(model_name, map_location=device)
        model.eval()

    env = gym.make(env_name, full_action_space=True)

    rng = np.random.default_rng()

    while not game_queue.empty():
        try:
            game = game_queue.get(False, None)
        except Empty:
            print("Game queue empty")
            return
        # Pick a random number of no-ops to perform
        no_ops = randint(0, args.no_op)
        no_ops_done = 0

        o = env.reset()
        r, d, i = (0.0, False, None)

        total_reward = 0
        total_frames = 0

        # Create a frame stack and fill it with zeros (black images)
        stack = []
        for _ in range(args.framestack):
            stack.append(np.zeros((args.width, args.height, 3), dtype=np.uint8))

        while True:
            if args.display:
                env.render()

            # Resize image
            img = Image.fromarray(o)
            img = img.resize((args.width, args.height), Image.BILINEAR)
            img = np.asarray(img)

            # Update the frame stack
            stack.insert(0, img)
            while len(stack) > args.framestack:
                stack.pop()

            # Make sure we have enough frames stacked
            if len(stack) != args.framestack:
                continue

            
            if args.merge:
                # Convert numpy arrays to images
                image_stack = map(Image.fromarray, stack)

                # Get lightest pixel values from the stack
                img = reduce(ImageChops.lighter, image_stack)

                np_stack = np.asarray(img, dtype=np.float32)
                np_stack = np.expand_dims(np_stack, axis=0)
            else:
                # Convert stack to numpy array with correct dimensions and type
                np_stack = np.concatenate(stack, axis=2)
                np_stack = np.expand_dims(np_stack, axis=0)
                np_stack = np_stack.astype(np.float32)

            # Normalize
            np_stack /= 255

            if no_ops_done < no_ops:
                # Send a no-op action if we haven't done enough no-ops yet
                o, r, d, i = env.step(0)
                no_ops_done += 1
            
            elif not args.random:
                prediction = model(torch.Tensor(np.swapaxes(np_stack, 1, 3)).to(device)).detach().cpu()
                prediction = F.softmax(prediction, dim=1)

                if args.action == "argmax":
                    prediction = np.argmax(prediction)
                elif args.action == "sampling":
                    # Perform a weighted selection from the indices
                    prediction = np.array(prediction[0])
                    p = prediction/np.sum(prediction)
                    prediction = rng.choice(list(range(len(prediction))), p=p)

                o, r, d, i = env.step(prediction)
            elif args.random:
                o, r, d, i = env.step(np.random.randint(18))

            total_reward += r
            total_frames += 1

            # Stop evaluation if game reaches terminal state or
            # maximum number of frames is exceeded
            if d or total_frames > args.max_frames:
                reward_queue.put(total_reward)
                break
        
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

        # Queue for holding the rewards from processes
        rewards = multiprocessing.Manager().Queue(1000000)
        
        # Queue for holding remaining game IDs
        games = multiprocessing.Manager().Queue(1000000)
        for i in range(args.games):
            games.put(i)

        procs = []

        # Start processes
        # Using threads doesn't work as the OpenAI Atari gym crashes if run
        # from multiple threads at the same time. Processes work fine though.
        for i in range(args.processes):
            proc = Process(target=play_game, args=(model, args.env, games, rewards, i))
            proc.start()
            procs.append(proc)

        print("Processes started")

        # Wait for processes to finish
        for k, proc in enumerate(procs):
            print("Waiting to join process #{}".format(k))
            proc.join()
            print("Joined process #{}".format(k))

        print("Processes joined")

        # Collect results from processes
        with open(results_file, "w") as f:
            rewards_list = []
            while not rewards.empty():
                r = rewards.get()
                rewards_list.append(r)
                f.write("{}\n".format(r))
                print(r)
            
            if len(rewards_list) <= 1:
                avg = 0
                std = 0
                minim = 0
                maxim = 0
            else:
                avg = round(statistics.mean(rewards_list), 1)
                std = round(statistics.stdev(rewards_list), 1)
                minim = min(rewards_list)
                maxim = max(rewards_list)
            f.write("Avg: {}".format(avg))
            print("Avg: {}, std: {}, min: {}, max: {}".format(avg, std, minim, maxim))

if __name__ == "__main__":
    main()
