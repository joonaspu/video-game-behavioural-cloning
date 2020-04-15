# Behavioural cloning experiments with video games

Code used for the paper "[Benchmarking End-to-End Behavioural Cloning on Video Games](https://arxiv.org/abs/2004.00981)". See examples of the agents playing [here](https://www.youtube.com/watch?v=2SMLpnUEIPw).

Behavioural cloning experiments with [Atari 2600 games](https://github.com/openai/gym), [ViZDoom](https://github.com/mwydmuch/ViZDoom) and modern video games with [ViControl](https://github.com/joonaspu/ViControl).

The experiments can be run with the `run_*.sh` scripts.

## Atari experiments
The Atari 2600 experiments can be run with the `run_atari_head.sh`, `run_atarigc.sh` and `run_atari_delayed.sh` scripts. These scripts require the *Atari Grand Challenge* and *Atari-HEAD* datasets to be available in the paths pointed to by the `ATARI_GC_DIR` and `ATARI_HEAD_DIR` environment variables. The Atari Grand Challenge dataset can be downloaded at http://atarigrandchallenge.com/data and the Atari-HEAD dataset at https://zenodo.org/record/3451402.

The `ATARI_GC_DIR` environment variable should point at the root directory of the extracted dataset, which contains the `screens` and `trajectories` subdirectories.

The `ATARI_HEAD_DIR` environment variable should point at a directory that has subdirectories for each game (i.e. `montezuma_revenge`, `ms_pacman` and `space_invaders` in our experiments). The `.tar.bz2` archives inside each game's directory should also be extracted.

## ViZDoom experiments

To gather data from a human player:

```
mkdir -p doom_recordings
python3 record_vizdoom.py --config doom_scenarios/health_gathering_supreme.cfg --num-games 20 --output doom_recordings
python3 record_vizdoom.py --config doom_scenarios/deathmatch.cfg --num-games 10 --output doom_recordings
```

After this, `run_vizdoom.sh` will train models with behavioural cloning and evaluate their performance. Models and evaluation logs will appear under `experiments` directory.


## ViControl experiments

Before the ViControl scripts can be used, the ViControl binaries and `messages_pb2.py` must be copied to the `video_game_env` directory. The Windows binaries and `messages_pb2.py` can be downloaded from the [ViControl releases page](https://github.com/joonaspu/ViControl/releases). Binaries for other platforms can be built by following the instructions in the project README file.

To record data from game, you need the name of the process. E.g. to capture data from Super Hexagon (make sure `vicontrol_recordings` directory exists):

```
python3 .\record_vizdoom.py superhexagon.exe SuperHexagon ./vicontrol_recordings
```

After gathering data, recorded buttons have to be converted to with `utils/convert_recordings_to_multidiscrete.sh` script:

```
mkdir -p ./vicontrol_recordings/SuperHexagon/trajectories
./utils/convert_recordings_to_multidiscrete.sh SuperHexagon ./vicontrol_recordings/SuperHexagon/trajectories_pressed_buttons ./vicontrol_recordings/SuperHexagon/trajectories
```

Now you can train the behavioural cloning model. Note that you have to tweak resolution to be appropiate for the game (e.g. does not blur too much). This example uses the parameters used in the paper:

```
mkdir -p ./experiments/vicontrol_models
python3 train.py ./vicontrol_recordings SuperHexagon ./experiments/vicontrol_models/superhexagon --epochs 30 --workers 4 --framestack 1 --l2 0.00001 --save-freq 30 --json --width 95 --height 60 --actions 2 --fileformat jpg
```

Finally, you can use the model to play Super Hexagon. Note that you have to set the correct resolution again.

```
python3 .\play_vicontrol.py --process superhexagon.exe --game SuperHexagon ./experiments/vicontrol_models/superhexagon --width 95 --height 60
```

To add support for more games, modify `utils/game_keyboard_mapping.py` to list what buttons should be tracked.
