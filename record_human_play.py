#!/usr/bin/env python3
#
# record_human_play.py
# Script for recording human gameplay and storing
# screen images + actions

import argparse
import time
import os
import json

from video_game_env.connection import Connection

# Note on the recording:
# Human actions per request are button presses/mouse movements
# that happened between last and current request.
# I.e. Human actions come delayed by one frame. Record
# accordingly.

# This code records human player of arbritrary game,
# and records the images (as seen on screen), saved on
# disk along with actions and rewards for imitation learning.
# Actions and rewards are stored in a .json file in
#     [args.output]/trajectories_pressed_buttons/[env_name]/[timestamp].json
# and frames are stored in
#     [args.output]/screens/[env_name]/[timestamp]/
# as #.png images, where # is the timestep in the environment.

# Structure of the .json file:
#
# {
#   "steps": [
#       {"b": buttons pressed in step 0, "m": mouse movement in step 0, "t": time since start in ms},
#       {"b": buttons pressed in step 1, "m": mouse movement in step 1, "t": time since start in ms},
#       ...
#       For [num_images] - 1 frames (last image does not have an action)
#   ]
# }


parser = argparse.ArgumentParser("""Record humans playing video games.

Hotkeys:
    - Page Up + Q: Quit
    - Page Up + R: Start recording, or
                   stop and start new recording
    - Page Up + S: Stop recording
""")

parser.add_argument("--dont-start-binary", action="store_true",
                    help="Do not start the recorder binary.")
parser.add_argument("--binary", default="video_game_env/main",
                    help="Path to the recorder binary.")
parser.add_argument("-f", "--framerate", type=int, default=20,
                    help="At what FPS we should store experiences (default: 20)")
parser.add_argument("-q", "--quality", type=int, default=80,
                    help="JPEG compression quality (default: 80)")
parser.add_argument("process_name", type=str,
                    help="Name of process to be recorded.")
parser.add_argument("env_name", type=str,
                    help="Name to be used when storing samples.")
parser.add_argument("output", type=str,
                    help="Root directory for saved recordings.")


def finish_recording(recording_path, env_name, unique_id, data):
    """Store recorded data into a json file"""
    trajectory_file = os.path.join(
        recording_path,
        "trajectories_pressed_buttons",
        "{}".format(env_name),
        "{}.json".format(unique_id)
    )
    with open(trajectory_file, "w") as f:
        json.dump(data, f)


def start_recording(recording_path, env_name):
    """
    Create and initialize any directories/files
    for recording, and return unique
    ID for this recording (timestamp).
    """
    unique_id = str(int(time.time()))
    screens_dir = os.path.join(
        recording_path,
        "screens",
        "{}".format(env_name),
        unique_id
    )
    trajectories_dir = os.path.join(
        recording_path,
        "trajectories_pressed_buttons",
        "{}".format(env_name)
    )
    os.makedirs(screens_dir)
    os.makedirs(trajectories_dir, exist_ok=True)

    return unique_id, screens_dir


def main(args):
    c = Connection(
        start_binary=not args.dont_start_binary,
        binary_path=args.binary
    )

    record = False
    # ID for current recording directories
    recording_id = None
    # Directory where to save images
    image_directory = None
    # Actions and other metadata per frame. To be
    # stored in a JSON file.
    recorded_data = []
    recording_index = 0
    recording_start_time = None

    # Store previous response
    # for storing actions with one frame delay
    previous_response = None
    # Also store when last response happened
    previous_frame_time = None

    frame_time = None
    target_time_per_frame = 1.0 / args.framerate
    print("Ready to record (Page Up + r)...")
    # KeyboardInterrupt catch for saving
    # unsaved data.
    try:
        while True:
            frame_time = time.time()
            # TODO check that there is a frame
            c.req.get_keys = True
            c.req.get_mouse = True
            c.req.get_image = True
            c.req.quality = args.quality
            c.req.process_name = args.process_name
            response = c.send_request()

            # Hotkeys for Record, Stop and Quit
            if "page up" in response.pressed_keys:
                if "q" in response.pressed_keys:
                    # Make sure we do not discard
                    # any samples
                    if record:
                        finish_recording(
                            args.output,
                            args.env_name,
                            recording_id,
                            recorded_data
                        )
                    exit()
                if "r" in response.pressed_keys:
                    # If recording, save current frames.
                    # Make sure we have some frames recorded,
                    # because otherwise this triggers too soon
                    if record and recording_index > args.framerate:
                        finish_recording(
                            args.output,
                            args.env_name,
                            recording_id,
                            recorded_data
                        )
                        print("Saved {} frames".format(recording_index))
                    elif record and recording_index < args.framerate:
                        continue

                    if not record:
                        # Show helpful info
                        print("Recording started (Page Up + s to stop)...")
                        print("Or Page Up + r to save current frames.")

                    record = True
                    recorded_data = []
                    previous_response = None
                    previous_frame_time = None
                    recording_id = None
                    recording_index = 0
                    recording_start_time = time.time()
                    recording_id, image_directory = start_recording(
                        args.output,
                        args.env_name
                    )
                    continue
                elif "s" in response.pressed_keys:
                    if record:
                        record = False
                        finish_recording(
                            args.output,
                            args.env_name,
                            recording_id,
                            recorded_data
                        )
                        print("Recording done with {} frames".format(recording_index))

            # Store actions and current image
            if record:
                # Store image
                image = response.image
                with open(os.path.join(image_directory, "{}.jpg".format(recording_index)), "wb") as f:
                    f.write(image)
                recording_index += 1

                # If we had previous_response, store actions.
                # This will delay actions by one frame (to align them),
                # and also will cause one frame to be without actions (final)
                if previous_response:
                    x, y = previous_response.mouse.x, previous_response.mouse.y
                    pressed_keys = tuple(previous_response.pressed_keys)
                    # Get timing of previous frame (the timing when screenshot was taken.
                    # Actions happen between frames, not at the specific times)
                    recording_time_ms = int((previous_frame_time - recording_start_time) * 1000)
                    recorded_data.append({
                        "m": (x, y),
                        "b": pressed_keys,
                        # Time when frame was recorded
                        "t": recording_time_ms
                    })

                previous_frame_time = frame_time
                previous_response = response

            # Sleep between requests, aiming for
            # the desired framerate
            sleep_time = target_time_per_frame - time.time() + frame_time
            if sleep_time <= 0.0:
                # Using standard print so we know how often
                # we are missing frames
                print("[Warning] Can not keep up with the desired framerate.")
                sleep_time = 0.0
            else:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        # Save if recording
        if record:
            print("Saving current data to disk...")
            finish_recording(
                args.output,
                args.process_name,
                recording_id,
                recorded_data
            )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
