import tensorflow as tf
import numpy as np
import os
import time
from true_loss import TrueLoss

import math


class Optimizer:

    def __init__(self, config, dataset, model):
        self.config = config
        self.ds = dataset
        self.model = model

        self.saver = tf.train.Saver(max_to_keep=100)

        # a class used to measure the true loss values
        self.true_loss = TrueLoss(prediction_model=self.model,
                                  config=self.config)

    # conduct a first order optimization to train the model parameters
    def run(self):

        # create a tensorflow session
        with tf.Session() as sess:

            start_time = time.time()

            # pretrain the network
            self.initialize_model(sess=sess)

            # temporary variables
            log_freq = self.config["performance_epoch_frequency"]

            # run the optimization for a number of epochs, keep track of losses and gradient norms
            L_approx_avg, L_hat_avg = 0, 0
            grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg = 0, 0
            grad_norm_loss_unclipped_avg, grad_norm_loss_clipped_avg = 0, 0

            # a header for the log columns
            print('epoch_idx, L_hat_avg, L_approx_avg, L_test,'
                  + 'grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg,'
                  + 'grad_norm_loss_unclipped_avg, grad_norm_loss_clipped_avg,'
                  + 'elapsed_time')

            for epoch_idx in range(self.config["num_epochs"]):

                # update the loss network for a number of steps
                for k in range(self.config["steps_loss"]):
                    # draw a random batch of training instances
                    X_batch, Y_batch = self.ds.draw_train_batch(self.config["batch_size"])

                    # compute the true loss value of the prediction model trained so far
                    L_true = self.true_loss.compute_session(sess, X=X_batch, Y=Y_batch)

                    # avoid learning from an undefined true loss value
                    if math.isnan(L_true):
                        continue
                    else:
                        # update the parameters of the loss network
                        L_approx_batch, grad_norm_loss_unclipped_batch, grad_norm_loss_clipped_batch  = \
                            self.model.update_loss_model(sess=sess, X_batch=X_batch,
                                                                    Y_batch=Y_batch,
                                                                    L_batch_true=L_true)
                        L_approx_avg += L_approx_batch
                        grad_norm_loss_unclipped_avg += grad_norm_loss_unclipped_batch
                        grad_norm_loss_clipped_avg += grad_norm_loss_clipped_batch

                        # update the prediction network for a number of steps
                for k in range(self.config["steps_prediction"]):
                    # draw a random batch of training instances
                    X_batch, Y_batch = self.ds.draw_train_batch(self.config["batch_size"])
                    # update the parameters of the prediction network
                    L_hat_batch, grad_norm_pred_unclipped_batch, grad_norm_pred_clipped_batch = \
                        self.model.update_prediction_model(sess=sess, X_batch=X_batch, Y_batch=Y_batch)

                    L_hat_avg += L_hat_batch
                    grad_norm_pred_unclipped_avg += grad_norm_pred_unclipped_batch
                    grad_norm_pred_clipped_avg += grad_norm_pred_clipped_batch

                    # print progress logs at some frequency
                if epoch_idx % log_freq == 0:

                    if epoch_idx > 0:
                        L_hat_avg /= log_freq*self.config["steps_prediction"]
                        grad_norm_pred_unclipped_avg /= log_freq*self.config["steps_prediction"]
                        grad_norm_pred_clipped_avg /= log_freq*self.config["steps_prediction"]

                        if self.config["steps_loss"] > 0:
                            L_approx_avg /= log_freq*self.config["steps_loss"]
                            grad_norm_loss_unclipped_avg /= log_freq * self.config["steps_loss"]
                            grad_norm_loss_clipped_avg /= log_freq * self.config["steps_loss"]
                        else:
                            L_approx_avg = 0
                            grad_norm_loss_unclipped_avg = 0
                            grad_norm_loss_clipped_avg = 0
                    else:
                        L_hat_avg /= self.config["steps_prediction"]
                        grad_norm_pred_unclipped_avg /= self.config["steps_prediction"]
                        grad_norm_pred_clipped_avg /= self.config["steps_prediction"]

                        if self.config["steps_loss"] > 0:
                            L_approx_avg /= self.config["steps_loss"]
                            grad_norm_loss_unclipped_avg /= self.config["steps_loss"]
                            grad_norm_loss_clipped_avg /= self.config["steps_loss"]
                        else:
                            L_approx_avg = 0
                            grad_norm_loss_unclipped_avg = 0
                            grad_norm_loss_clipped_avg = 0

                    # compute the true loss on the test set
                    L_test = self.true_loss.compute_session(sess, X=self.ds.X_test, Y=self.ds.Y_test)

                    # print the epoch, the accumulated average Lhat since last printing step,
                    # the loss approximation term, the test loss, and the unclipped and clipped
                    # grad norms, as well as the elapsed time since the start of the optimization
                    print(epoch_idx, L_hat_avg, L_approx_avg, L_test,
                          grad_norm_pred_unclipped_avg, grad_norm_pred_clipped_avg,
                          grad_norm_loss_unclipped_avg, grad_norm_loss_clipped_avg,
                          time.time() - start_time, sep=',')

                    L_hat_avg, L_approx_avg = 0, 0

                    self.saver.save(sess,
                                    "./saved_models/" +
                                    self.config['loss'] +
                                    "_stratified=" + str(self.config['stratified_batch']) + ".ckpt",
                                    global_step=epoch_idx // log_freq)


    def initialize_model(self, sess):

        if self.config['init_universal_loss'] == True:

            # initialize the prediction variables randomly
            sess.run(tf.global_variables_initializer())

            # a saver for the loss variables
            self.loss_saver = tf.train.Saver(var_list=self.model.loss_variables,
                                             max_to_keep=100)

            checkpoint_file = os.path.join(self.config["universal_loss_folder"], self.config['loss'] + '.ckpt')

            print('Restoring loss params from', checkpoint_file)

            # restore the loss variables from the list
            self.loss_saver.restore(sess, checkpoint_file)



        else:
            sess.run(tf.global_variables_initializer())

