from argparse import ArgumentParser
import pickle
import json
import os

import matplotlib.pyplot as plt

PLOT_DIR = "plots"

if __name__ == "__main__":
    parser = ArgumentParser("Plot Keras history object.")
    parser.add_argument("file", type=str,
                        help="Path to the pickled Keras history object.")
    parser.add_argument("--show", action="store_true",
                        help="Show the figure on screen.")
    parser.add_argument("--save", action="store_true",
                        help="Save the figure on disk.")
    parser.add_argument("--pickle", action="store_true",
                        help="History file is stored as a pickle instead of JSON.")

    args = parser.parse_args()

    if args.pickle:
        with open(args.file, "rb") as f:
            history = pickle.load(f)
    else:
        with open(args.file, "r") as f:
            history = json.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("{}".format(args.file))

    # Plot accuracy
    axs[0].plot(history["accuracy"], label="accuracy")
    axs[0].set_title("Model accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    # Plot loss
    axs[1].plot(history["loss"], label="loss")
    axs[1].set_title("Training loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")

    # Display and/or save the figure
    if args.save:
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        file_name = os.path.basename(os.path.normpath(args.file))
        fig.savefig(os.path.join(PLOT_DIR, "{}.png".format(file_name)))
    if args.show:
        fig.show()