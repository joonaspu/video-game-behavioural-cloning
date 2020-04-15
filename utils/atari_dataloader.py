import os
from random import randrange, uniform, choice, shuffle
from functools import lru_cache, reduce
from time import perf_counter
from io import BytesIO
import multiprocessing
import json

from PIL import Image, ImageEnhance, ImageChops
from PIL.ImageDraw import Draw
import numpy as np
import lz4.frame

ATARI_W, ATARI_H = 160, 210

def compress_to_bytes(compress=True, **kwargs):
    """
    Compress a dict of numpy arrays with .savez
    and return the bytes.

    Parameters:
        compress: If True, compress the bytes using
                  compression algorithm (using LZ4)
        kwargs: Numpy arrays that will be stored.
                Fed directly to numpy.savez
    """
    bytes_buffer = BytesIO()
    return_bytes = None
    if compress:
        np.savez(bytes_buffer, **kwargs)
        # Compress using LZ4
        return_bytes = lz4.frame.compress(
            bytes_buffer.getvalue(),
        )
    else:
        np.savez(bytes_buffer, **kwargs)
        return_bytes = bytes_buffer.getvalue()
    return return_bytes


def decompress_to_arrays(array_bytes, compress=True):
    """
    Decompress bytearray back to numpy arrays. Inverse
    of `compress_to_bytes`

    Parameters:
        bytearray: Bytearray to be decompressed
        compress: If True, bytes were compressed with
                  LZ4 and require decompressing.
    """
    if compress:
        # Decompress LZ4
        array_bytes = lz4.frame.decompress(
            array_bytes,
        )
    bytes_buffer = BytesIO(array_bytes)
    return_arrays = np.load(bytes_buffer)
    return return_arrays

class AtariDataLoader():
    """Keras Sequence where the elements are batches from the Atari dataset"""

    def __init__(self, directory, game, batch_size=32, stack=3, controls=18,
                 size=(84, 84), percentile=None, top_n=None, augment=False,
                 preload=False, merge=False, dqn=False, json=False,
                 fileformat="png", action_delay=0):

        self.dir = directory
        self.game = game
        self.fileformat = fileformat

        self.batch_size = batch_size
        self.stack = stack
        self.controls = controls
        self.size = size

        self.traj_path = os.path.join(self.dir, "trajectories", self.game)
        self.screen_path = os.path.join(self.dir, "screens", self.game)

        self.all_trajs = self._get_trajectory_list()
        self.n_traj = len(self.all_trajs)

        self.augment = augment
        self.merge = merge

        # If this is true, then next state, reward and terminal are included
        # in the returned tuples as well
        # Otherwise only current state and action are included
        self.dqn = dqn

        # This should be true if the dataset is stored as JSON instead of CSV
        self.json = json

        self.action_delay = action_delay

        # Get trajectory lengths and scores
        self.traj_len = []
        self.scores = []

        self.total_len = 0
        for i in range(self.n_traj):
            self.traj_len.append(self._get_samples_in_trajectory(i))
            self.scores.append(self._get_sample_score(i))
            self.total_len += self.traj_len[i]

        self.traj_len_all = self.traj_len[:]

        # Filter by percentile
        if percentile is not None:
            # Filter trajectories that fall into the desired percentile
            p = np.percentile(self.scores, percentile)
            top = filter(lambda x: x[1] >= p, zip(range(self.n_traj), self.scores))

            # traj_len will contain (id, length) tuples for filtered trajectories
            self.traj_len = list(map(lambda x: (x[0], self.traj_len[x[0]]), top))

            # Sum up the lengths of the filtered trajectories
            self.total_len = sum(map(lambda x: x[1], self.traj_len))

        # Filter the top n games
        elif top_n is not None:
            # Sort by score and select top games
            top = sorted(zip(range(self.n_traj), self.scores),
                         key=lambda x: x[1], 
                         reverse=True)[:top_n]

            # traj_len will contain (id, length) tuples for filtered trajectories
            self.traj_len = list(map(lambda x: (x[0], self.traj_len[x[0]]), top))

            # Sum up the lengths of the filtered trajectories
            self.total_len = sum(map(lambda x: x[1], self.traj_len))

        else:
            # If no percentile was given..
            # ..traj_len will contain (id, length) tuples for all trajectories
            self.traj_len = list(zip(range(self.total_len), self.traj_len))

        # Preload data to cache
        self.cache = []

        if preload:
            for batch in range(len(self)):
                data = self.get_batch(batch)
                b = compress_to_bytes(img=data[0], action=data[1])
                self.cache.append(b)
                print("Cached {}/{}".format(batch, len(self)))
            print("Preload done!")


    def _get_image_stacked(self, traj, id, augments=None):
        """Returns time-stacked or merged images from
        the given trajectory and sample ID,
        depending on the value of self.merge
        """
        stack = []
        shape = None

        # Collect frames as numpy arrays to a list
        for i in range(self.stack):
            ix = id - i
            if ix >= 0:
                stack.insert(0, self._get_image(traj, ix, augments))
                if shape is None:
                    shape = stack[0].shape
            else:
                stack.insert(0, np.zeros(shape, dtype=np.uint8))

        if self.merge:
            # Convert numpy arrays to images
            stack = map(Image.fromarray, stack)

            # Get lightest pixel values from the stack
            img = reduce(ImageChops.lighter, stack)
            
            return np.asarray(img, dtype=np.uint8)
        else:
            return np.concatenate(stack, axis=2)

    @lru_cache(maxsize=int(1e6))
    def _get_image(self, traj, id, augments=None):
        """Returns image from the given trajectory and sample ID as
        a numpy array
        """
        traj_name = self.all_trajs[traj]

        filename = "{}.{}".format(id, self.fileformat)
        path = os.path.join(self.dir, "screens", self.game, traj_name, filename)

        img = Image.open(path)
        img.load()
        
        if augments is not None:
            if "shadow" in augments: # (alpha, x, y, w, h)
                # Draw a randomly placed rectangle on the image
                draw = Draw(img, "RGBA")
                rect_color = (0, 0, 0, augments["shadow"][0])
                rect_w = augments["shadow"][3]
                rect_h = augments["shadow"][4]
                rect_x = augments["shadow"][1]
                rect_y = augments["shadow"][2]
                draw.rectangle([rect_x - rect_w/2, rect_y - rect_h/2, rect_x + rect_w/2, rect_y + rect_h/2], rect_color)

            if "brightness" in augments:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(augments["brightness"])
            if "rotate" in augments:
                img = img.rotate(augments["rotate"])
            if "shear" in augments:
                raise NotImplementedError
            if "tx" in augments and "ty" in augments: # Shift
                img = ImageChops.offset(img, xoffset=augments["tx"], yoffset=augments["ty"])
            if "zx" in augments and "zy" in augments: # Zoom
                raise NotImplementedError
            if "flip" in augments and augments["flip"]:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize(self.size, Image.BILINEAR)
        img = np.asarray(img, dtype=np.uint8)
        
        return img

    def _flip_controls(self, control, game=None):
        """Flips the controls horizontally, i.e. switches left and right buttons.
        Since qbert has diagonal movement, flipping the controls is
        more complicated, and the 'game' parameter must be set to 'qbert'."""

        controls = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT',
                    'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE',
                    'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',
                    'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

        control_name = controls[control]
        new_control_name = control_name

        if game == "qbert":
            if control_name == "UP":
                new_control_name = "LEFT"
            elif control_name == "RIGHT":
                new_control_name = "DOWN"
            elif control_name == "LEFT":
                new_control_name = "UP"
            elif control_name == "DOWN":
                new_control_name = "RIGHT"
            elif control_name == "UPRIGHT":
                new_control_name = "DOWNLEFT"
            elif control_name == "DOWNLEFT":
                new_control_name = "UPRIGHT"
            elif control_name == "UPFIRE":
                new_control_name = "LEFTFIRE"
            elif control_name == "RIGHTFIRE":
                new_control_name = "DOWNFIRE"
            elif control_name == "LEFTFIRE":
                new_control_name = "UPFIRE"
            elif control_name == "DOWNFIRE":
                new_control_name = "RIGHTFIRE"
            elif control_name == "UPRIGHTFIRE":
                new_control_name = "DOWNLEFTFIRE"
            elif control_name == "DOWNLEFTFIRE":
                new_control_name = "UPRIGHTFIRE"
            
        else:
            control_name = controls[control]
            if "RIGHT" in control_name:
                new_control_name = control_name.replace("RIGHT", "LEFT")
            elif "LEFT" in control_name:
                new_control_name = control_name.replace("LEFT", "RIGHT")
            
        return controls.index(new_control_name)

    @lru_cache(maxsize=128)
    def _get_data_lines(self, traj):
        traj_name = "{}.txt".format(self.all_trajs[traj])

        with open(os.path.join(self.traj_path, traj_name)) as f:
            return f.read().splitlines()

    def _get_data(self, traj, id, flip=False):
        """Returns a list with the following contents:
            [frame, reward, score, terminal, action, last]
        """

        lines = self._get_data_lines(traj)
        num_frames = len(lines) - 2
        data = lines[id + 2].split(",")
        # Strip whitespace
        data = [s.strip() for s in data]
        # Convert to correct data type
        for i in range(5):
            if i == 3:
                data[i] = True if data[i].lower() == "true" else False
            else:
                data[i] = int(float(data[i]))

        # Flip controls horizontally
        if flip:
            data[4] = self._flip_controls(data[4], game="qbert" if self.game == "qbert" else None)

        # Check if this is the last frame
        if id >= num_frames - 1:
            last = 1
        else:
            last = 0

        data.append(last)
        return list(data)

    @lru_cache(maxsize=128)
    def _get_json(self, traj):
        traj_name = "{}.json".format(self.all_trajs[traj])

        with open(os.path.join(self.traj_path, traj_name)) as f:
            return json.load(f)

    def _get_data_json(self, traj, id, flip=False):
        """Returns a list with the following contents:
            [frame, reward, score, terminal, action, last]
        """

        data = self._get_json(traj)["steps"]
        last_id = len(data) - 1
        data = data[id]

        return [id, data["r"], 0, id == last_id, data["a"], id == last_id]

    def _get_trajectory_list(self):
        """Returns a sorted list of all trajectory names"""
        # Trajectory data
        traj_datas = set(map(lambda x: x.split(".")[0], os.listdir(self.traj_path)))

        # Trajectory screens
        traj_screens = set(os.listdir(self.screen_path))

        # Intersection of both sets to make sure we have
        # both screens AND data
        trajs = traj_datas & traj_screens

        return list(sorted(trajs, key=int))

    def _get_num_of_trajectories(self):
        """Returns the number of trajectories in this dataset"""
        trajectories = os.listdir(
            os.path.join(self.dir, "trajectories", self.game)
        )

        return len(trajectories)

    def _get_sample_score(self, traj):
        """Returns the final score of the given trajectory ID"""
        if self.json:
            traj_name = "{}.json".format(self.all_trajs[traj])

            with open(os.path.join(self.traj_path, traj_name)) as f:
                steps = json.load(f)["steps"]
                score = 0
                for step in steps:
                    score += step["r"]
                return int(score)
        else:
            traj_name = "{}.txt".format(self.all_trajs[traj])

            with open(os.path.join(self.traj_path, traj_name)) as f:
                lines = f.read().splitlines()
                # Get score from last line
                return int(float(lines[-1].split(",")[2]))

    def _get_samples_in_trajectory(self, traj):
        """Returns the number of samples in the given trajectory ID"""
        if self.json:
            traj_name = "{}.json".format(self.all_trajs[traj])

            with open(os.path.join(self.traj_path, traj_name)) as f:
                lines = len(json.load(f)["steps"])
                return lines
        else:
            traj_name = "{}.txt".format(self.all_trajs[traj])

            with open(os.path.join(self.traj_path, traj_name)) as f:
                lines = f.read().splitlines()
                # Number of samples is the ID of last line plus one
                return int(lines[-1].split(",")[0]) + 1

    def _get_index_traj_and_sample(self, index):
        """Returns the corresponding trajectory ID and sample ID for the
        given index"""
        total = 0
        for i, t_len in self.traj_len:
            if index < total + t_len:
                return i, index - total
            total += t_len

    def __len__(self):
        return int(self.total_len / self.batch_size)
    
    def get_batch(self, samples):
        batch_x = []
        batch_y = []
        if self.dqn:
            batch_x_next = []
            batch_reward = []
            batch_done = []

        for sample in samples:
            traj, ix = self._get_index_traj_and_sample(sample)
            flip = False
            if self.augment:
                flip = choice([True, False])
                augments = {
                    "shadow": ( randrange(128, 255), # (alpha, x, y, w, h)
                                uniform(0, ATARI_W), uniform(0, ATARI_H),
                                uniform(10, 100), uniform(10, 100)),
                    "brightness": uniform(0.5, 1.5),
                    "rotate": uniform(-2, 2),
                    #"shear": uniform(-2, 2),
                    "tx": randrange(-5, 5),
                    "ty": randrange(-5, 5),
                    #"zx": uniform(0.95, 1.05),
                    #"zy": uniform(0.95, 1.05),
                    "flip": flip,
                }
            else:
                augments = None

            if self.json:
                data = self._get_data_json(traj, ix, flip)
            else:
                data = self._get_data(traj, ix, flip)

            if self.action_delay != 0:
                # Add action delay to the index
                action_ix = ix + self.action_delay

                # If the index goes out of bounds, move it back to the
                # first/last element of the trajectory
                if action_ix >= self.traj_len_all[traj]:
                    action_ix = self.traj_len_all[traj] - 1

                if action_ix < 0:
                    action_ix = 0

                # Get delayed action and put it in the list
                if self.json:
                    delayed_data = self._get_data_json(traj, action_ix, flip)
                else:
                    delayed_data = self._get_data(traj, action_ix, flip)
                data[4] = delayed_data[4]

            # If this is the last frame but it's not marked as terminal,
            # skip it
            if self.dqn and data[5] == 1 and data[3] == 0:
                continue

            batch_x.append(self._get_image_stacked(traj, ix, augments))
            if self.json:
                batch_y.append(np.array([data[4]]))
            else:
                batch_y.append(np.eye(self.controls)[data[4]])
            if self.dqn:
                batch_reward.append(data[1])
                batch_done.append(data[5])
                if not data[5]:
                    # Add next frame
                    batch_x_next.append(
                        self._get_image_stacked(traj, ix + 1, augments)
                    )
                else:
                    # Add a black frame
                    batch_x_next.append(
                        np.zeros(batch_x[0].shape, dtype=np.uint8)
                    )

        if self.dqn == False:
            # (state, action)
            return (
                np.array(batch_x, dtype=np.uint8),
                np.array(batch_y, dtype=np.uint8)
            )
        else:
            # (state, action, next_state, reward, done)
            return (
                np.array(batch_x, dtype=np.uint8),
                np.array(batch_y, dtype=np.uint8),
                np.array(batch_x_next, dtype=np.uint8),
                np.array(batch_reward),
                np.array(batch_done, dtype=np.uint8)
            )

class AtariDataLoaderProcess(multiprocessing.Process):
    """Process that runs a single AtariDataLoader instance"""

    def __init__(self, request_queue, response_queue, dataloader_args):
        self.loader = AtariDataLoader(**dataloader_args)
        self.request_queue = request_queue
        self.response_queue = response_queue

        super().__init__()

    def __len__(self):
        return len(self.loader)

    def run(self):
        while True:
            # Request a batch by taking a list of sample IDs from the queue
            response = self.loader.get_batch(self.request_queue.get())
            self.response_queue.put(response)
        
class MultiprocessAtariDataLoader():
    """Creates multiple dataloader processes and serves data from them
    as an iterator
    
    Note: The iterator can return batches in any order, but is guaranteed
    to return every batch exactly once.
    """

    def __init__(self, dataloader_args, workers):
        super().__init__()

        self.request_queue = multiprocessing.Manager().Queue()
        self.queue = multiprocessing.Queue(maxsize=workers)

        loader = AtariDataLoader(**dataloader_args)
        self.batch_size = loader.batch_size
        self.sample_length = loader.total_len # Total number of samples
        self.length = len(loader) # Number of batches
        self.shape = loader._get_image(0, 0).shape

        self.loaders = []
        # Create and start child processes
        for i in range(workers):
            new_loader = AtariDataLoaderProcess(
                self.request_queue, self.queue, dataloader_args
            )
            self.loaders.append(new_loader)
        for i in self.loaders:
            i.start()   
    
    def stop(self):
        for i in self.loaders:
            i.terminate() 

    def __len__(self):
        return self.length

    def __next__(self):
        if self.iters < self.length:
            response = self.queue.get()
            self.iters += 1
            return response
        else:
            raise StopIteration

    def __iter__(self):
        self.iters = 0

        # Generate random sequence of samples
        samples = list(range(self.sample_length))
        shuffle(samples)

        # Fill request queue
        for i in range(self.length):
            if i % 1000 == 0:
                print("Adding batch {} to queue".format(i))
            batch = []
            for _ in range(self.batch_size):
                batch.append(samples.pop())
            self.request_queue.put(batch)
        
        return self
