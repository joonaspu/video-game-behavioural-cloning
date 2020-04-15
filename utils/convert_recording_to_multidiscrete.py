#!/usr/bin/env python3
#
# convert_recording_to_multidiscrete.py
# Take a game recording done with record_human_play.py,
# and convert their trajectory files (.json) to ones
# like in ViZDoom, which can used when training BC.
#

import argparse
import time
import os
import json

from game_keyboard_mapping import KEY_MAPPING

# Convert .json files from this
# {
#   "steps": [
#       {"b": buttons pressed in step 0, "m": mouse movement in step 0, "t": time since start in ms},
#       {"b": buttons pressed in step 1, "m": mouse movement in step 1, "t": time since start in ms},
#       ...
#       For [num_images] - 1 frames (last image does not have an action)
#   ]
# }
#
# To this
# {
#   "allowed_buttons": [LIST OF RECORDED BUTTONS],
#   "steps": [
#       {"a": actions pressed in step 0, "r": 0.0, "t": time since start in ms},
#       {"a": actions pressed in step 1, "r": 0.0, "t": time since start in ms},
#       ...
#   ]
#


parser = argparse.ArgumentParser("Convert recording files for training")
parser.add_argument("game", type=str, choices=list(KEY_MAPPING.keys()),
                    help="Game for which this mapping is done.")
parser.add_argument("input", type=str,
                    help="Input JSON file.")
parser.add_argument("output", type=str,
                    help="Output JSON file.")


def main(args):
    input_data = None
    with open(args.input) as f:
        input_data = json.load(f)

    key_mapping = KEY_MAPPING[args.game]

    # Use first buttons in the keymapping as
    # buttons-to-be-pressed when played
    button_representatives = [buttons[0] for buttons in key_mapping]
    new_data = {
        "allowed_buttons": button_representatives,
        "steps": None
    }
    new_steps = []
    for step in input_data:
        new_step = {
            "r": 0.0,
            "t": step["t"],
        }

        pressed_buttons = step["b"]
        new_action = [
            int(any(
                [button in pressed_buttons for button in buttons]
            ))
            for buttons in key_mapping
        ]
        new_step["a"] = new_action
        new_steps.append(new_step)

    new_data["steps"] = new_steps

    with open(args.output, "w") as f:
        json.dump(new_data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
