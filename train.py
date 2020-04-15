from time import perf_counter
from argparse import ArgumentParser
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.atari_dataloader import MultiprocessAtariDataLoader
from utils.atari_head_dataloader import MultiprocessAtariHeadDataLoader

from utils.networks import Mnih2015

if __name__ == "__main__":
    parser = ArgumentParser("Train PyTorch models to do imitation learning.")
    parser.add_argument("input_directory", type=str,
                        help="Path to directory with recorded gameplay.")
    parser.add_argument("game", type=str,
                        help="Name of the game to use for training.")
    parser.add_argument("model", nargs="?", type=str,
                        help="Path of the file where model will be saved.") 
    parser.add_argument("--actions", type=int, default=18,
                        help="Number of actions")       
    parser.add_argument("--framestack", type=int, default=3,
                        help="Number of frames to stack")
    parser.add_argument("--merge", action="store_true",
                        help="Merge stacked frames into one image.")
    parser.add_argument("--width", "-x", type=int, default=84,
                        help="Width of the image")
    parser.add_argument("--height", "-y", type=int, default=84,
                        help="Height of the image")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes to use for the dataloader.")
    parser.add_argument("--l2", type=float, default="0.00001",
                        help="L2 regularization weight.")
    parser.add_argument("--percentile", type=int,
                        help="The top q-percentile of samples to use for training.")
    parser.add_argument("--top-n", type=int,
                        help="The top n number of samples to use for training.")
    parser.add_argument("--save-freq", type=int, default=1,
                        help="Number of epochs between checkpoints.")
    parser.add_argument("--augment", action="store_true",
                        help="Use image augmentation.")
    parser.add_argument("--preload", action="store_true",
                        help="Preload image data to memory.")
    parser.add_argument("--atari-head", action="store_true",
                        help="Use the Atari-HEAD dataloader.")
    parser.add_argument("--action-delay", type=int, default=0,
                        help="How many frames to delay the actions by.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Don't use CUDA")
    parser.add_argument("--json", action="store_true",
                        help="Dataset is stored as JSON")
    parser.add_argument("--fileformat", type=str, default="png",
                        help="Postfix of the image files to be loaded")

    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mnih2015(
        (args.width, args.height),
        3 if args.merge else 3*args.framestack,
        args.actions
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2)

    dataloader_args = {
        "directory": args.input_directory,
        "game": args.game,
        "stack": args.framestack,
        "batch_size": args.batch,
        "size": (args.width, args.height),
        "percentile": args.percentile,
        "top_n": args.top_n,
        "augment": args.augment,
        "preload": args.preload,
        "merge": args.merge,
        "json": args.json,
        "action_delay": args.action_delay,
        "fileformat": args.fileformat
    }

    # Note: if new dataloader arguments are added, make sure they work with
    #       both loaders, or if they don't, remove them with 'del' below
    if args.atari_head:
        del dataloader_args["game"]
        del dataloader_args["json"]
        del dataloader_args["fileformat"]
        gen = MultiprocessAtariHeadDataLoader(dataloader_args, args.workers)
    else:
        gen = MultiprocessAtariDataLoader(dataloader_args, args.workers)
    shape = gen.shape

    history = dict()
    history["loss"] = []
    history["accuracy"] = []

    for epoch in range(1, args.epochs + 1):
        print("Starting epoch {}".format(epoch))
        model.train()
        start = perf_counter()

        # Accuracy
        correct = 0
        total = 0

        # Loss
        loss_sum = 0
        loss_num = 0

        for batch, data in enumerate(gen):
            # Convert data to correct format
            x = torch.Tensor(np.swapaxes(data[0], 1, 3)).to(device) / 255
            if args.json:
                # Drop unnecessary axis
                y = torch.Tensor(data[1]).to(device)[:, 0, :]
            else:
                y = torch.argmax(torch.Tensor(data[1]).to(device), 1).long()

            optimizer.zero_grad()

            # Get model output
            output = model(x)

            # Calculate loss
            if args.json:
                loss = F.binary_cross_entropy_with_logits(output, y)
            else:
                loss = F.cross_entropy(output, y)

            # Add loss to epoch statistics
            loss_sum += loss
            loss_num += 1

            # Calculate accuracy and add to epoch statistics
            if args.json:
                correct += 0 # TODO
            else:
                correct += output.argmax(1).eq(y).sum()

            total += len(y)
            
            # Backpropagate loss
            loss.backward()
            optimizer.step()

            # Print statistics
            if batch % 100 == 0:
                end = perf_counter()
                accuracy = float(correct) / float(total)
                loss = loss_sum / loss_num
                print("Epoch {} - {}/{}: loss: {}, acc: {} ({} s/batch)".format(
                    epoch,
                    batch,
                    len(gen),
                    loss,
                    accuracy,
                    (end - start) / 100)
                )
                start = perf_counter()

        # Save statistics
        accuracy = float(correct) / float(total)
        loss = loss_sum / loss_num

        history["accuracy"].append(float(accuracy))
        history["loss"].append(float(loss))

        with open(args.model + "-history.json", "w") as f:
            json.dump(history, f)

        # Save model
        if args.model is not None and epoch % args.save_freq == 0:
            filename = "{}_{}.pt".format(args.model, epoch)
            print("Saving {}".format(filename))
            torch.save(model, filename)

        print("Finished epoch {}".format(epoch))
    
    gen.stop()
