import os
import io
from argparse import ArgumentParser
from random import randint
from functools import reduce
import time

import numpy as np
import gym
import torch
from PIL import Image, ImageChops

from video_game_env.connection import Connection
from utils.game_keyboard_mapping import KEY_MAPPING

from record_human_play import start_recording, finish_recording

parser = ArgumentParser("Play a game over ViControl with a trained model. [Page Up] + p to play/pause.")
parser.add_argument("model", type=str,
                    help="Path of the file(s) where the model will be loaded from.")
parser.add_argument("--game", type=str, required=True, choices=list(KEY_MAPPING.keys()),
                    help="Name of the game to be played (for button-set).")
parser.add_argument("--process", type=str, required=True,
                    help="Process from which to capture images.")
parser.add_argument("--framestack", type=int, default=1,
                    help="Number of frames to stack (must match the number used in model)")
parser.add_argument("--framerate", type=int, default=20,
                    help="How often agent controls the game")
parser.add_argument("--width", "-x", type=int, default=84,
                    help="Width of the image")
parser.add_argument("--height", "-y", type=int, default=84,
                    help="Height of the image")
parser.add_argument("--action", type=str, default="sampling",
                    choices=["sampling", "argmax"],
                    help="Use random sampling or argmax to pick actions.")
parser.add_argument("--no-cuda", action="store_true",
                    help="Disable CUDA")
parser.add_argument("--output", type=str, default=None,
                    help="If provided, output frames and actions to these folders")
parser.add_argument("--random", action="store_true",
                    help="Ignore model and just pick random actions.")
# ViControl parameters
parser.add_argument("--dont-start-binary", action="store_true",
                    help="Do not start the recorder binary.")
parser.add_argument("--binary", default="video_game_env/main",
                    help="Path to the recorder binary.")
parser.add_argument("-q", "--quality", type=int, default=80,
                    help="JPEG compression quality (default: 80)")

args = parser.parse_args()


if args.no_cuda:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = None
    # Do not bother loading a model if we use a random agent
    if not args.random:
        model = torch.load(args.model, map_location=device)
        model.eval()

    c = Connection(
        start_binary=not args.dont_start_binary,
        binary_path=args.binary
    )

    buttons = [buttons[0] for buttons in KEY_MAPPING[args.game]]
    num_buttons = len(buttons)

    # Switch to toggle when the agent plays
    is_playing = False

    target_time_per_frame = 1.0 / args.framerate
    frame_time = None

    # Create a frame stack and fill it with zeros (black images)
    stack = []
    for _ in range(args.framestack):
        stack.append(np.zeros((args.width, args.height, 3), dtype=np.float32))

    print("Ready to play (Page Up + p)...")

    # For storing images.
    recording_id = None
    image_directory = None
    recorded_data = []
    recording_index = 0

    while True:
        frame_time = time.time()
        c.req.allow_user_override = True
        c.req.get_keys = True
        c.req.get_image = True
        c.req.quality = args.quality
        c.req.process_name = args.process
        response = c.send_request()

        if "page up" in response.pressed_keys:
            if "p" in response.pressed_keys and not is_playing:
                is_playing = True
                print("Starting to play (stop with Page Up + s)")
                print("Currently playing: " + str(is_playing))
                for _ in range(args.framestack):
                    stack.append(
                        np.zeros(
                            (args.width, args.height, 3),
                            dtype=np.uint8
                        )
                    )
                if args.output is not None:
                    recording_index = 0
                    recording_id, image_directory = start_recording(
                        args.output,
                        args.game
                    )
                    recorded_data = []
            elif "s" in response.pressed_keys and is_playing:
                print("Stopped playing. Start with Page Up + p")
                is_playing = False
                if args.output is not None:
                    finish_recording(
                        args.output,
                        args.game,
                        recording_id,
                        recorded_data
                    )

        if is_playing:
            # Resize image
            img = Image.open(io.BytesIO(response.image))
            img = img.resize((args.width, args.height), Image.BILINEAR)
            img = np.asarray(img, dtype=np.float32)

            # Update the frame stack
            stack.insert(0, img)
            while len(stack) > args.framestack:
                stack.pop()

            # Make sure we have enough frames stacked
            if len(stack) != args.framestack:
                continue

            # Convert stack to numpy array with correct dimensions and type
            np_stack = np.concatenate(stack, axis=2)
            np_stack = np.expand_dims(np_stack, axis=0)
            np_stack = np_stack.astype(np.float32)

            # Normalize
            np_stack /= 255

            # Get prediction
            prediction = None
            if not args.random:
                prediction = model(torch.Tensor(np.swapaxes(np_stack, 1, 3)).to(device)).detach().cpu()[0]
                prediction = torch.sigmoid(prediction).numpy()

                # Convert prediction to a list of {0, 1} values for each control
                prediction = (np.random.random(size=prediction.shape) < prediction).astype(np.int)
                prediction = prediction.tolist()
            else:
                prediction = np.random.randint(2, size=num_buttons).tolist()

            # Set buttons down or up, depending on the prediction
            for i in range(len(buttons)):
                if prediction[i] == 1:
                    c.req.press_keys.append(buttons[i])
                else:
                    c.req.release_keys.append(buttons[i])
            c.req.get_image = False
            c.req.get_keys = False
            _ = c.send_request()

            # If recording, save frame and buttons
            if args.output is not None:
                image = response.image
                with open(os.path.join(image_directory, "{}.jpg".format(recording_index)), "wb") as f:
                    f.write(image)
                recorded_data.append({
                    "b": [buttons[i] for i in range(len(buttons)) if prediction[i]]
                })
                recording_index += 1

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

if __name__ == "__main__":
    main()
