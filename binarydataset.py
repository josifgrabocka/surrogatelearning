import numpy as np
import os


class BinaryDataset:

    # the constructor loads the data
    def __init__(self, folder_path, config):

        self.config = config

        self.X_train = np.load(os.path.join(folder_path, 'train_features.npy'))
        self.Y_train = np.load(os.path.join(folder_path, 'train_labels.npy'))

        self.X_test = np.load(os.path.join(folder_path, 'test_features.npy'))
        self.Y_test = np.load(os.path.join(folder_path, 'test_labels.npy'))

        self.num_train_instances, self.num_features = self.X_train.shape
        self.num_test_instances = self.X_test.shape[0]

        print('Features:', 'Train', self.X_train.shape, 'Test', self.X_test.shape,
              'Target: Train', self.Y_train.shape, 'Test', self.Y_test.shape)

        print('Mean target',
              'Train', np.mean(self.Y_train),
              'Test', np.mean(self.Y_test))

        if self.config["stratified_batch"]:
            self.pos_train_idxs = np.where(self.Y_train == 1)[0]
            self.neg_train_idxs = np.where(self.Y_train == 0)[0]

            self.pos_test_idxs = np.where(self.Y_test == 1)[0]
            self.neg_test_idxs = np.where(self.Y_test == 0)[0]

    # draw a random batch of training instances
    def draw_train_batch(self, batch_size):

        batch_indices = None

        # if the batch should be stratified, then draw according to the specified ratio of positive instances
        if self.config["stratified_batch"]:
            num_pos = int(batch_size*self.config['stratification_pos_ratio'])
            num_neg=batch_size-num_pos

            pos_idxs = np.random.choice(self.pos_train_idxs, num_pos)
            neg_idxs = np.random.choice(self.neg_train_idxs, num_neg)
            batch_indices = np.concatenate([pos_idxs, neg_idxs])
            np.random.shuffle(batch_indices)

        # otherwise draw the batch entirely random
        else:
            # but ensure at least one positive instance is drawn
            while True:
                batch_indices = np.random.choice(np.arange(0,self.num_train_instances), batch_size)
                if np.sum(self.Y_train[batch_indices, :]) > 0:
                    break

        return self.X_train[batch_indices, :], self.Y_train[batch_indices, :]


    # draw a random batch of testing instances
    def draw_test_batch(self, batch_size):

        batch_indices = None

        if self.stratified_batch:
            num_pos = int(batch_size * self.config['stratification_pos_ratio'])
            num_neg = batch_size - num_pos

            pos_idxs = np.random.choice(self.pos_test_idxs, num_pos)
            neg_idxs = np.random.choice(self.neg_test_idxs, num_neg)

            batch_indices = np.concatenate([pos_idxs, neg_idxs])
            np.random.shuffle(batch_indices)

        else:
            # ensure at least one positive instance drawn
            while True:
                batch_indices = np.random.choice(np.arange(0,self.num_test_instances), batch_size)
                if np.sum(self.Y_test[batch_indices, :]) > 0:
                    break

        return self.X_test[batch_indices, :], self.Y_test[batch_indices, :]
