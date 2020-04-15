import json
import re
import os
import statistics

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def human_normalized_score(score, random, human, stdev=None):
    norm_score = 100 * (score - random) / (human - random)
    if stdev is not None:
        upper = 100 * ((score + stdev) - random) / (human - random)
        lower = 100 * ((score - stdev) - random) / (human - random)
        return norm_score, upper - norm_score, norm_score - lower
    else:
        return norm_score, 0, 0

def figure_nodelay_atari():
    with open("results.json") as f:
        results = json.load(f)

    atari_games = [
        "Ms. Pac-Man",
        "Video Pinball",
        "Q*bert",
        "Montezuma's Revenge",
        "Space Invaders"
    ]

    _, axs = plt.subplots(len(atari_games), 1, sharex=True, figsize=(6, 8))

    for k, game in enumerate(atari_games):
        labels = []
        means = []
        stdevs_low = []
        stdevs_high = []

        for dataset in ["Top 5%", "Top 50%", "All", "Atari-HEAD"]:
            if dataset not in results["bc"][game]:
                continue

            labels.append("{}".format(dataset))

            mean = results["bc"][game][dataset]["mean"]
            stdev = results["bc"][game][dataset]["stdev"]

            # Normalize mean and stdev so that
            # 0 = random agent, 100 = human player
            norm_mean, norm_upper, norm_lower = human_normalized_score(
                mean,
                results["random"][game]["mean"],
                results["human"][game]["mean"],
                stdev
            )

            means.append(norm_mean)
            stdevs_low.append(norm_lower)
            stdevs_high.append(norm_upper)
            stdevs = [stdevs_low, stdevs_high]

        # Y axis ordering top-down
        axs[k].invert_yaxis()

        # Set Y tick names
        axs[k].set_yticks(range(len(labels)))
        axs[k].set_yticklabels(labels)

        # Set tick label size
        axs[k].tick_params(axis="both", which="major")

        # Hide X tick labels from all but the lowest subplot
        if k != len(atari_games) - 1:
            plt.setp(axs[k].get_xticklabels(), visible=False)
        else:
            axs[k].set_xlabel("% of human score")

        # Set the same X limits for all subplots
        axs[k].set_xlim(left=-10, right=35)
        
        # Set title of subplots to the game name
        axs[k].set_title(game, fontsize="medium")

        # Draw the (horizontal) bar plot
        axs[k].barh(range(len(labels)), means, xerr=stdevs)

        # Add gridlines to the X axis
        axs[k].grid(b=True, which='major', axis="x", color='#999999',
                    linestyle='-', linewidth=0.25)

        # Set grid to be below everything else
        axs[k].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure_atari.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.savefig("figure_atari.png", dpi=400, bbox_inches='tight', pad_inches=0)

def figure_nodelay():
    with open("results.json") as f:
        results = json.load(f)

    games = results["bc"].keys()

    atari_games = [
        "Ms. Pac-Man",
        "Video Pinball",
        "Q*bert",
        "Montezuma's Revenge",
        "Space Invaders"
    ]

    # List of all games except for the Atari games listed above
    games = [game for game in games if game not in atari_games]

    _, ax = plt.subplots(1, figsize=(6, 3.25))

    labels = []
    means = []
    stdevs_low = []
    stdevs_high = []

    for game in games:
        labels.append("{}".format(game))

        mean = results["bc"][game]["All"]["mean"]
        stdev = results["bc"][game]["All"]["stdev"]

        # Normalize mean and stdev so that
        # 0 = random agent, 100 = human player
        norm_mean, norm_upper, norm_lower = human_normalized_score(
            mean,
            results["random"][game]["mean"],
            results["human"][game]["mean"],
            stdev
        )

        means.append(norm_mean)
        stdevs_low.append(norm_lower)
        stdevs_high.append(norm_upper)

    stdevs = [stdevs_low, stdevs_high]

    # Y axis ordering top-down
    ax.invert_yaxis()

    # Set Y tick names
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)

    # Set tick label size
    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("% of human score")

    # Draw the (horizontal) bar plot
    ax.barh(range(len(labels)), means, xerr=stdevs)

    # Add gridlines to the X axis
    ax.grid(b=True, which='major', axis="x", color='#999999',
                linestyle='-', linewidth=0.25)

    # Set grid to be below everything else
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure_all.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.savefig("figure_all.png", dpi=400, bbox_inches='tight', pad_inches=0)

def figure_delay():
    with open("results.json") as f:
        results = json.load(f)

    _, axs = plt.subplots(2, 5, figsize=(12, 5), sharex=True)

    # Colors for the delay values (blue -> red)
    coolwarm = cm.get_cmap("coolwarm", 9)
    colors = [coolwarm(x) for x in np.linspace(0, 1, 9)]

    for row in range(2):
        if row == 0:
            dataset = "atarigc_95"
        else:
            dataset = "atarihead"
        games = results["delay_{}".format(dataset)].keys()
        for k, game in enumerate(games):
            game_name = game.replace("\n(Atari-HEAD)", "") \
                            .replace("\n(Atari GC)", "")
            labels = []
            means = []
            stdevs_low = []
            stdevs_high = []
            for delay in ["-100", "-10", "-5", "-2", "0", "2", "5", "10", "100"]:
                mean = results["delay_{}".format(dataset)][game][delay]["mean"]
                stdev = results["delay_{}".format(dataset)][game][delay]["stdev"]

                # Normalize mean so that 0 = random agent, 100 = human player
                norm_mean, norm_upper, norm_lower = human_normalized_score(
                    mean,
                    results["random"][game_name]["mean"],
                    results["human"][game_name]["mean"],
                    stdev
                )

                means.append(norm_mean)
                stdevs_low.append(norm_lower)
                stdevs_high.append(norm_upper)
                labels.append(delay)

            axs[row, k].bar(range(len(labels)), means, yerr=[stdevs_low, stdevs_high],
                    width=1.0, color=colors)

            # Add labels to X axis ticks
            axs[row, k].set_xticks(range(len(labels)))
            axs[row, k].set_xticklabels(labels, rotation="vertical")

            # Set tick label size
            axs[row, k].tick_params(axis="x", which="major")

            if k == 0:
                axs[row, k].set_ylabel("% of human score")
            axs[row, k].set_title(game, fontsize="medium")

    plt.tight_layout()
    plt.savefig("figure_delay.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.savefig("figure_delay.png", dpi=400, bbox_inches='tight', pad_inches=0)

def figure_learning():
    def get_avg_from_file(file_path):
        with open(file_path) as f:
            avg_line = f.readlines()[-1]
            match = re.match(r"Avg: (.*)", avg_line)
            return float(match.group(1))

    def get_stdev_from_file(file_path):
        values = get_datapoints_from_file(file_path)

        return statistics.stdev(values)

    def get_datapoints_from_file(file_path):
        with open(file_path) as f:
            lines = f.readlines()
            values = []
            for line in lines:
                try:
                    values.append(float(line))
                except ValueError:
                    pass

            return values

    with open("results/space_invaders_all_2-history.json", "r") as f:
        history = json.load(f)

    _, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Plot accuracy
    #axs[0].plot(history["accuracy"], label="accuracy")
    #axs[0].set_title("Model accuracy")
    #axs[0].set_ylabel("Accuracy")
    #axs[0].set_xlabel("Epoch")

    # Plot loss
    axs[0].plot(history["loss"], label="loss")
    axs[0].set_title("Training loss", fontsize="medium")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(range(10))
    axs[0].set_xticklabels([1,2,3,4,5,6,7,8,9,10])

    # Regex for finding the result files
    # 1st group: name
    # 2nd group: epoch number
    repeat = 2
    r = re.compile(r"(.*)_{}_([0-9]{{1,4}})\.pt\.txt".format(repeat))

    files = []
    path = os.path.normpath("results/")

    # Find matching files
    for entry in os.listdir(path):
        full_entry = os.path.join(path, entry)
        if os.path.isfile(full_entry):
            match = r.match(entry)
            if match is not None and match.group(1) == "space_invaders_all":
                epoch = int(match.group(2))
                files.append((
                    epoch,
                    get_avg_from_file(full_entry),
                    get_stdev_from_file(full_entry),
                    get_datapoints_from_file(full_entry)
                ))

    # Sort the file list by epoch
    files.sort(key=lambda x: x[0])

    x, y, yerr, points = zip(*files)
    x = list(x)
    y = list(y)
    yerr = list(yerr)

    for epoch, entry, stdev, _ in files:
        print("{}: {} (std {})".format(epoch, entry, stdev))

    for i, v in enumerate(x):
        for _y in points[i]:
            plt.scatter(v, _y, marker="_", c="#00000028", linewidths=1)

    axs[1].errorbar(x, y, yerr=yerr)
    axs[1].set_title("Evaluation score", fontsize="medium")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Score")
    axs[1].set_xticks([1,2,3,4,5,6,7,8,9,10])
    axs[1].set_xticklabels([1,2,3,4,5,6,7,8,9,10])

    plt.tight_layout()
    plt.savefig("figure_learning.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.savefig("figure_learning.png", dpi=400, bbox_inches='tight', pad_inches=0)

if __name__ ==  "__main__":
    figure_nodelay_atari()
    figure_nodelay()
    figure_delay()
    figure_learning()
