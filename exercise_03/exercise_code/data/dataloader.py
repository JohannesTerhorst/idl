"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        # The order in which we visit the samples, as a list of positions.
        # shuffle=True:  a random order, e.g. [7, 2, 9, 0, ...]
        # shuffle=False: the normal order     [0, 1, 2, 3, ...]
        # We only shuffle these positions, never the dataset itself, so no
        # sample is loaded before it is actually needed.
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))

        # i is the position where the current batch starts.
        # With batch_size=3 it takes the values 0, 3, 6, 9, ...
        for i in range(0, len(self.dataset), self.batch_size):
            # Not enough samples left for a full batch? With drop_last=True
            # we throw these leftovers away and stop here.
            if self.drop_last and i + self.batch_size > len(self.dataset):
                break

            # Step 1: load the samples of this batch, and only these.
            # min(...) makes sure we don't run past the end of the dataset
            # when the last batch is smaller than batch_size.
            # Result: a list of dicts, e.g. [{"data": 2}, {"data": 4}, {"data": 6}]
            batch_fetch = []
            for j in range(i, min(i + self.batch_size, len(self.dataset))):
                batch_fetch.append(self.dataset[indices[j]])

            # Step 2: merge the list of dicts into ONE dict that holds a list
            # per key, e.g. {"data": [2, 4, 6]}.
            # This works for any keys, e.g. "image" and "label" for CIFAR-10.
            batch = {}
            for sample in batch_fetch:
                for key, value in sample.items():
                    # First time we see this key: start an empty list for it.
                    if key not in batch:
                        batch[key] = []
                    batch[key].append(value)

            # Step 3: turn every list into a numpy array, which is faster to
            # compute with, e.g. {"data": array([2, 4, 6])}.
            for key in batch:
                batch[key] = np.array(batch[key])

            # Step 4: hand this batch to the caller (e.g. a for loop) and pause.
            # The function continues right here when the next batch is
            # requested, so only one batch is held in memory at a time.
            yield batch


    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################
        # Number of FULL batches. // is integer division and rounds down,
        # e.g. 50 samples // 3 per batch = 16 full batches (2 samples left over).
        length = len(self.dataset) // self.batch_size

        # % gives the remainder of the division, i.e. the leftover samples
        # (50 % 3 = 2). If there are leftovers and drop_last is False, they
        # form one extra, smaller batch at the end, so we count one more.
        # With drop_last=True the leftovers are thrown away, so 16 stays 16.
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            length += 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
