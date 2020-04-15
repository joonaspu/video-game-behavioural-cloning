from argparse import ArgumentParser
import csv
import os
import statistics

import matplotlib.pyplot as plt

HIST_DIR = "histograms"

if __name__ == "__main__":
    parser = ArgumentParser("Print stats about behavior cloning datasets.")
    parser.add_argument("--atari-head", type=str,
                        help="Path to the Atari-HEAD meta_data.csv.")

    parser.add_argument("--atari-gc", type=str,
                        help="Path to the Atari grand challenge dataset.")

    args = parser.parse_args()

    if args.atari_head:
        stats = dict()
        with open(args.atari_head) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["GameName"] not in stats:
                    stats[row["GameName"]] = {"scores": [], "samples": []}

                stats[row["GameName"]]["scores"].append(float(row["highest_score"]))
                stats[row["GameName"]]["samples"].append(int(row["total_frame"]))

        for game in stats:
            print("{} mean: ${} \\pm {}$, min: {}, max: {}, len: {}, total samples: {}, mean samples: ${} \\pm {}$".format(
                game,
                round(statistics.mean(stats[game]["scores"]), 1),
                round(statistics.stdev(stats[game]["scores"]), 1),
                min(stats[game]["scores"]),
                max(stats[game]["scores"]),
                len(stats[game]["scores"]),
                sum(stats[game]["samples"]),
                round(statistics.mean(stats[game]["samples"]), 1),
                round(statistics.stdev(stats[game]["samples"]), 1)
            ))
            plt.figure()
            plt.hist(stats[game]["scores"])
            #plt.title("Atari-HEAD dataset: {}".format(game))
            plt.xlabel("Score")
            plt.ylabel("Episodes")
            plt.savefig(os.path.join(HIST_DIR, "{}_head.pdf".format(game)))

    if args.atari_gc:
        trajectories_path = os.path.join(args.atari_gc, "trajectories")
        games = os.listdir(trajectories_path)

        game_scores = dict()

        for game in games:
            scores = []
            samples = []
            game_path = os.path.join(trajectories_path, game)
            trajectories = os.listdir(game_path)
            for trajectory in trajectories:
                traj_path = os.path.join(game_path, trajectory)
                with open(traj_path) as f:
                    last_line = f.readlines()[-1]
                    final_score = int(last_line.split(",")[2])
                    final_frame = int(last_line.split(",")[0])
                    scores.append(final_score)
                    samples.append(final_frame + 1)

            game_scores[game] = scores

            print("{}: mean: ${} \\pm {}$, min: {}, max: {}, len: {}, total samples: {}, mean samples: ${} \\pm {}$".format(
                game,
                round(statistics.mean(scores), 1),
                round(statistics.stdev(scores), 1),
                min(scores),
                max(scores),
                len(scores),
                sum(samples),
                round(statistics.mean(samples), 1),
                round(statistics.stdev(samples), 1)
            ))
            plt.figure()
            plt.hist(scores)
            #plt.title("Atari Grand Challenge dataset: {}".format(game))
            plt.xlabel("Score")
            plt.ylabel("Episodes")
            plt.savefig(os.path.join(HIST_DIR, "{}_gc.pdf".format(game)))

    if args.atari_gc and args.atari_head:
        _, axs = plt.subplots(1, 5, figsize=(6, 2))

        for k, game in enumerate([("mspacman", "ms_pacman", "Ms. Pac-Man"),
                ("pinball", None, "Video Pinball"),
                ("qbert", None, "Q*bert"),
                ("revenge", "montezuma_revenge", "Montezuma's\nRevenge"),
                ("spaceinvaders", "space_invaders", "Space\nInvaders")]
            ):

            #plt.figure(figsize=(6, 3))
            axs[k].hist(
                game_scores[game[0]] if game[1] is None else [game_scores[game[0]], stats[game[1]]["scores"]],
                density=True,
                label=["Atari GC"] if game[1] is None else ["Atari GC", "Atari-HEAD"],
                bins=20,
                histtype="stepfilled",
                alpha=0.5
            )
            if k == 0:
                axs[k].legend(fontsize=6)
            axs[k].set_title(game[2], fontsize=10)
            axs[k].set_xlabel("Score")
            axs[k].tick_params(axis="x", which="major", labelsize=7)
            if k == 0:
                axs[k].set_ylabel("Density")
            axs[k].set_yticks([])

        plt.tight_layout(pad=0.1, w_pad=0.02)

        plt.savefig(os.path.join(HIST_DIR, "atari_histogram.pdf"))
        plt.savefig(os.path.join(HIST_DIR, "atari_histogram.png"))
