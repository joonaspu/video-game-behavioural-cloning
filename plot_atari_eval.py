from argparse import ArgumentParser
import statistics
import os
import re

import matplotlib.pyplot as plt

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


PLOT_DIR = "plots"

if __name__ == "__main__":
    parser = ArgumentParser("Plot Atari evaluation results.")
    parser.add_argument("path", type=str,
                        help="Path to the directory where the result files are loaded from.")
    parser.add_argument("name", type=str,
                        help="Name of the evaluation to plot.")
    parser.add_argument("--show", action="store_true",
                        help="Show the figure on screen.")
    parser.add_argument("--save", action="store_true",
                        help="Save the figure on disk.")
    parser.add_argument("--noplot", action="store_true",
                        help="Do not do plotting.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeated experiments.")

    args = parser.parse_args()

    averages = []

    for repeat in range(1, args.repeats + 1):
        # Regex for finding the result files
        # 1st group: name
        # 2nd group: epoch number
        r = re.compile(r"(.*)_{}_([0-9]{{1,4}})\.pt\.txt".format(repeat))

        files = []
        path = os.path.normpath(args.path)

        # Find matching files
        for entry in os.listdir(path):
            full_entry = os.path.join(path, entry)
            if os.path.isfile(full_entry):
                match = r.match(entry)
                if match is not None and match.group(1) == args.name:
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

        # Average of the final three
        avrg_of_last_three = statistics.mean(y[-3:])
        averages.append(avrg_of_last_three)
        print("Average of final three eval points: ", avrg_of_last_three)

        if args.noplot:
            continue

        plt.figure()
        plt.rcParams["figure.figsize"] = (8, 6)

        for i, v in enumerate(x):
            for _y in points[i]:
                plt.scatter(v, _y, marker="_", c="#00000028", linewidths=1)

        plt.errorbar(x, y, yerr=yerr)
        plt.title("{}_{}, max: {}: avrg[-3:]: {}".format(
            args.name,
            repeat,
            round(max(y), 2),
            round(avrg_of_last_three, 2)
        ))

        if args.save:
            if not os.path.exists(PLOT_DIR):
                os.makedirs(PLOT_DIR)
            file_name = os.path.basename(os.path.normpath("{}_{}".format(args.name, repeat)))
            plt.savefig(os.path.join(PLOT_DIR, "{}.png".format(file_name)))
        if args.show:
            plt.show()

    print("{}: ${} \pm {}$".format(
        args.name,
        round(statistics.mean(averages), 1),
        round(statistics.stdev(averages), 1)
    ))
