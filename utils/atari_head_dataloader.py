import os
from functools import reduce, lru_cache
import multiprocessing
from random import randrange, uniform, choice, shuffle

import numpy as np
from PIL import Image, ImageChops

# TODO: Some features (e.g. percentile, top_n and augment) are not implemented.
#       Lots of code has been copied from atari_dataloader.py with minor
#       changes, should look into reusing the same classes/methods instead.

class AtariHeadDataloader():
    def __init__(self, directory, batch_size=32, stack=3, controls=18,
                 size=(84, 84), percentile=None, top_n=None, augment=False,
                 preload=False, merge=False, dqn=False, action_delay=0,
                 print_stats=False):

        self.batch_size = batch_size
        self.stack = stack
        self.controls = controls
        self.size = size
        self.merge = merge
        self.dqn = dqn
        self.action_delay = action_delay

        self.directory = directory

        self.all_trajs = self._get_trajectory_list()
        self.n_traj = len(self.all_trajs)
        
        self.traj_len = []
        for traj in range(len(self.all_trajs)):
            self.traj_len.append(self._get_samples_in_trajectory(traj))

        self.total_len = sum(self.traj_len)

    def _get_trajectory_list(self):
        """Returns a sorted list of all trajectory names"""
        names = os.listdir(self.directory)
        names = list(filter(lambda x: x.endswith(".txt"), names))
        names = list(map(lambda x: x[:-4], names))
        names = sorted(names)
        return names

    def _get_samples_in_trajectory(self, traj):
        """Returns the number of samples in the given trajectory ID"""
        lines = self._get_data_lines(traj)
        return len(lines) - 1

    def _get_index_traj_and_sample(self, index):
        """Returns the corresponding trajectory ID and sample ID for the
        given index"""
        total = 0
        for i, t_len in enumerate(self.traj_len):
            if index < total + t_len:
                return i, index - total
            total += t_len

    def _get_frame_id(self, traj, index):
        """Returns the frame_id field for the given trajectory and index"""
        lines = self._get_data_lines(traj)
        return lines[index + 1].split(",")[0]

    def _get_image_stacked(self, traj, id):
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
                stack.insert(0, self._get_image(traj, ix))
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

    def _get_image(self, traj, index):
        traj_name = self.all_trajs[traj]
        frame_id = self._get_frame_id(traj, index)

        filename = "{}.png".format(frame_id)
        path = os.path.join(self.directory, traj_name, filename)

        img = Image.open(path)
        img.load()
        img = img.resize(self.size, Image.BILINEAR)
        img = np.asarray(img, dtype=np.uint8)
        
        return img

    @lru_cache(maxsize=128)
    def _get_data_lines(self, traj):
        traj_name = "{}.txt".format(self.all_trajs[traj])

        with open(os.path.join(self.directory, traj_name)) as f:
            return f.read().splitlines()

    def _get_data(self, traj, id):
        """Returns a list with the following contents:
            [frame, reward, score, terminal, action, last]
        """

        lines = self._get_data_lines(traj)
        num_frames = len(lines) - 1
        data = lines[id + 1].split(",")[:6] # We don't need the gaze data
        # Strip whitespace
        data = [s.strip() for s in data]
        # Convert to correct data type

        # Score (can be null)
        try:
            data[2] = int(data[2])
        except ValueError:
            data[2] = -1

        # Reward (can be null)
        try:
            data[4] = int(data[4])
        except ValueError:
            data[4] = 0
        
        # Action (can be null)
        try:
            data[5] = int(data[5]) 
        except ValueError:
            data[5] = 0

        # Check if this is the last frame
        if id >= num_frames - 1:
            last = 1
        else:
            last = 0

        data.append(last)
        return [id, data[4], data[2], False, data[5], last]

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
            data = self._get_data(traj, ix)

            if self.action_delay != 0:
                # Add action delay to the index
                action_ix = ix + self.action_delay

                # If the index goes out of bounds, move it back to the
                # first/last element of the trajectory
                if action_ix >= self.traj_len[traj]:
                    action_ix = self.traj_len[traj] - 1

                if action_ix < 0:
                    action_ix = 0

                # Get delayed action and put it in the list
                delayed_data = self._get_data(traj, action_ix)
                data[4] = delayed_data[4]

            # If this is the last frame but it's not marked as terminal,
            # skip it
            if self.dqn and data[5] == 1 and data[3] == 0:
                continue

            batch_x.append(self._get_image_stacked(traj, ix))
            batch_y.append(np.eye(self.controls)[data[4]])
            if self.dqn:
                batch_reward.append(data[1])
                batch_done.append(data[5])
                if not data[5]:
                    # Add next frame
                    batch_x_next.append(
                        self._get_image_stacked(traj, ix + 1)
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
        self.loader = AtariHeadDataloader(**dataloader_args)
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
        
class MultiprocessAtariHeadDataLoader():
    """Creates multiple dataloader processes and serves data from them
    as an iterator
    
    Note: The iterator can return batches in any order, but is guaranteed
    to return every batch exactly once.
    """

    def __init__(self, dataloader_args, workers):
        super().__init__()

        self.request_queue = multiprocessing.Manager().Queue()
        self.queue = multiprocessing.Queue(maxsize=workers)

        loader = AtariHeadDataloader(**dataloader_args)
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