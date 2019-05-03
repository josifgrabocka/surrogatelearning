import tensorflow as tf
import numpy as np
from true_loss import TrueLoss

import math


class UniversalLossOptimizer:

    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.loss_saver = tf.train.Saver(var_list=model.loss_variables,
                                         max_to_keep=100)

        # a class used to measure the true loss values
        self.true_loss = TrueLoss(prediction_model=self.model,
                                  config=self.config)

    # conduct a first order optimization to train the model parameters
    def run(self):

        # create a tensorflow session
        with tf.Session() as sess:

            # randomly set all trainable parameters
            sess.run(tf.global_variables_initializer())

            # temporary variables
            log_freq = self.config["performance_epoch_frequency"]

            # run the optimization for a number of epochs
            L_approx_total = 0
            for epoch_idx in range(self.config["num_epochs"]):

                # update the loss network for a number of steps
                for k in range(self.config["steps_loss"]):

                    # generate random estimations and ground truths
                    Y_hat = np.random.normal(loc=0.0, scale=1.0, size=(self.config['batch_size'], 1))
                    Y_true = np.random.binomial(n=1, p=0.5, size=(self.config['batch_size'], 1))

                    # compute the true loss value of the prediction model trained so far
                    L_true = self.true_loss.compute(Y_hat=np.reshape(Y_hat, newshape=(-1)),
                                                    Y=np.reshape(Y_true, newshape=(-1)))

                    # avoid learning from an undefined true loss value
                    if math.isnan(L_true):
                        continue
                    else:
                        # update the parameters of the loss network
                        _, L_approx_batch = sess.run(fetches=[self.model.loss_model_update, self.model.loss_loss],
                                           feed_dict={self.model.Y_true: Y_true,
                                                      self.model.L_true: L_true,
                                                      self.model.Y_hat: Y_hat,
                                                      self.model.is_training: True})

                        L_approx_total += L_approx_batch

                # print progress logs at some frequency
                if epoch_idx % log_freq == 0:

                    if epoch_idx > 0:
                        L_approx_total /= log_freq*self.config["steps_loss"]
                    else:
                        L_approx_total /= self.config["steps_loss"]

                    # compute the true loss on the test set
                    print('Train', epoch_idx, L_approx_total, sep=',')
                    L_approx_total = 0

                    self.loss_saver.save(sess,
                                    "./saved_loss_models/" +
                                         self.config['loss'] + ".ckpt",
                                         global_step=epoch_idx // log_freq)
