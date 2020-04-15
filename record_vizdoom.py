#!/usr/bin/env python3
#
# record_vizdoom.py
#
# Tool for recording human gameplay in ViZDoom
# and storing the trajectories
#

from time import sleep, time
import os
from argparse import ArgumentParser
import json

import numpy as np
from cv2 import resize, imwrite
import vizdoom as vzd

import keyboard

# This code plays ViZDoom scenarios for human player,
# and records the images (as seen on screen), saved on
# disk along with actions and rewards for imitation learning.
# Actions and rewards are stored in a .json file in
#     [args.output]/trajectories/vizdoom_[config_name]/[timestamp].json
# and frames are stored in
#     [args.output]/screens/vizdoom_[config_name]/[timestamp]/
# as #.png images, where # is the timestep in the environment.

# Structure of the .json file:
#
# {
#   "allowed_buttons": [LIST OF RECORDED BUTTONS],
#   "steps": [
#       {"a": actions pressed in step 0, "r": reward in step 0, "t": time since start in ms},
#       {"a": actions pressed in step 1, "r": reward in step 1, "t": time since start in ms},
#       ...
#   ]
# }


# Hardcoded image resolution for imitation
# learning purposes on light hardware.
# We do resizing here already to save on
# space. (Width, height)
RESOLUTION = (160, 120)

# Mapping from ViZDoom buttons to
# keyboard buttons
BUTTON_MAPPING = {
    vzd.Button.TURN_LEFT: "left arrow",
    vzd.Button.TURN_RIGHT: "right arrow",
    vzd.Button.MOVE_FORWARD: "up arrow",
    vzd.Button.MOVE_BACKWARD: "down arrow",
    vzd.Button.SPEED: "left shift",
    vzd.Button.ATTACK: "left ctrl",
}

def save_episode(frames, actions, rewards, timestamps, allowed_buttons, args):
    """
    Save one episode of gameplay into args.output,
    putting frames as .png files under
        "args.output/screens/[timestamp]"
    directory and rest information in
        "args.output/trajectories/[timestamp].json"
    """
    config_name = os.path.basename(args.config).replace(".cfg", "")
    unique_id = str(int(time()))
    screens_dir = os.path.join(
        args.output,
        "screens",
        "vizdoom_{}".format(config_name),
        unique_id
    )
    trajectories_dir = os.path.join(
        args.output,
        "trajectories",
        "vizdoom_{}".format(config_name)
    )
    os.makedirs(screens_dir)
    os.makedirs(trajectories_dir, exist_ok=True)

    json_dict = {
        "allowed_buttons": list(map(str, allowed_buttons)),
        "steps": list(map(
            lambda x: {"a": x[0], "r": x[1], "t": x[2]}, zip(actions, rewards, timestamps)
        ))
    }

    with open(os.path.join(trajectories_dir, unique_id + ".json"), "w") as f:
        f.write(json.dumps(json_dict))

    for i, frame in enumerate(frames):
        filepath = os.path.join(screens_dir, "{}.png".format(i))
        # cv2 saves as BGR, but frames are in RGB
        imwrite(filepath, frames[i][:, :, ::-1])


def get_keyboard_actions(available_buttons):
    """
    Check which of the available buttons are pressed down
    according to BUTTON_MAPPING, and return a vizdoom
    action
    """
    ret = []
    for available_button in available_buttons:
        if keyboard.is_pressed(BUTTON_MAPPING[available_button]):
            ret.append(1)
        else:
            ret.append(0)
    return ret


def main(args):
    game = vzd.DoomGame()

    game.load_config(args.config)

    game.set_window_visible(True)
    
    if args.sync is not None:
        game.set_mode(vzd.Mode.PLAYER)
    else:
        game.set_mode(vzd.Mode.SPECTATOR)

    allowed_buttons = game.get_available_buttons()

    try:
        game.init()
    except Exception as e:
        # Check if buffer mismatch error
        if "size mismatch" in str(e):
            print("Could not run ViZDoom at desired resolution. " +
                  "Try changing the resolution in the config file " +
                  "(e.g. RES_1920X1080 works on 1080p monitors)")
            exit(1)
        else:
            raise e

    for i in range(args.num_games):
        print("Episode #" + str(i + 1))

        game.new_episode()

        states = []
        actions = []
        rewards = []
        episode_times = []
        start_time = time()

        step_ctr = 0
        while not game.is_episode_finished():
            state = game.get_state()
            episode_times.append(int(
                (time() - start_time) * 1000
            ))
            
            if args.sync is not None:
                # Manual control: Use separate keyboard
                # tracking to get the actions
                sleep(args.sync)
                
                actions = get_keyboard_actions(allowed_buttons)
                game.make_action(actions, args.rate)
                last_action = actions
            else:
                # Let humans play at same rate as
                # recorded, so they can not change their
                # actions between and have hidden effects
                # like that.
                game.advance_action(args.rate)
                last_action = game.get_last_action()

            reward = game.get_last_reward()

            # Only save every Nth frame, to save on
            # space
            frame = state.screen_buffer
            # ViZDoom gives images as
            # CHW, turn to HWC
            frame = frame.transpose([1, 2, 0])
            frame = resize(frame, RESOLUTION)
            states.append(frame.astype(np.uint8))
            # TODO what if we have continuous actions
            actions.append(list(map(int, last_action)))
            rewards.append(reward)
            step_ctr += 1

        save_episode(states, actions, rewards, episode_times, allowed_buttons, args)

    game.close()


if __name__ == "__main__":
    parser = ArgumentParser("Record human gameplay in ViZDoom.")
    parser.add_argument("--config",
                        required=True,
                        type=str,
                        help="Path to the configuration file of the scenario.")
    parser.add_argument("--num-games",
                        default=10,
                        type=int,
                        help="How many games will be played.")
    parser.add_argument("--rate",
                        default=2,
                        type=int,
                        help="How many frames between saving an experience.")
    parser.add_argument("--output",
                        default="data",
                        help="Path where recorded gameplay should be stored.")
    parser.add_argument("--sync",
                        default=None,
                        type=float,
                        help="Play game in 'synchronized' mode, where human player has this many seconds to select an action.")
    args = parser.parse_args()
    main(args)
