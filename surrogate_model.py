import numpy as np
import tensorflow as tf

# the prediction model, which infers the targets of a batch, plus
# the loss model, which estimates the loss of a batch
class SurrogateModel:

    def __init__(self, config):

        # pointer to the dataset and the configuration dictionary
        self.config = config

        with tf.variable_scope('PlaceHolders'):
            # the features
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.config['num_features']), name='X')
            # the ground truth target
            self.Y_true = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Y_true')
            # the groud truth loss terms
            self.L_true = tf.placeholder(dtype=tf.float32, shape=(), name='L_true')
            # a flag to indicate training mode
            self.is_training = tf.placeholder(tf.bool)

        # tensor operations for the estimated target and the estimated loss
        self.Y_hat, self.L_hat= None, None
        # the list of prediction variables and loss variabels
        self.pred_variables, self.loss_variables = None, None
        # update ops
        self.prediction_model_update, self.loss_model_update = None, None

    # create the models
    def create_model(self):
        # create the network architecture
        self.create_prediction_model()
        self.create_loss()
        self.create_update_rules()

    # create the network that predicts the target variable
    def create_prediction_model(self):

        # create the prediction network
        with tf.variable_scope('PredictionNetwork'):
            # batch normalize the feature tensor
            layer = self.X
            for idx, n in enumerate(self.config['prediction_layers']):
                # create a dense layer in the prediction model with a Leaky Relu activation
                layer = tf.layers.dense(inputs=layer,
                                        activation=tf.nn.leaky_relu,
                                        units=n,
                                        name='Dense'+str(idx))
                # add a batch normalization layer after the dense layer
                layer = tf.layers.batch_normalization(inputs=layer,
                                                      training=self.is_training,
                                                      name='BatchNorm'+str(idx))
                # add a dropout after the batch normalization
                layer = tf.layers.dropout(inputs=layer,
                                          rate=self.config['dropout_rate'],
                                          training=self.is_training,
                                          name='Dropout'+str(idx))

            # the estimated target for a binary decision problem (one output neuron)
            self.Y_hat = tf.layers.dense(inputs=layer,
                                    activation=None,
                                    units=1,
                                    name='Y_hat')

            # store the variables of the prediction network
            self.pred_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='PredictionNetwork')

    # create the loss network
    def create_loss(self):

        with tf.variable_scope('LossNetwork'):
            # concatenate the estimated and true target variable to feed it as
            # the input of the loss estimating network
            layer = tf.concat([self.Y_hat, self.Y_true],
                          axis=1,
                          name='ConcatenatedLossInput')

            # create the network that will output the latent embedding of each pair of
            # estimated and true targets
            for idx, n in enumerate(self.config['loss_embedding_layers']):
                # the dense layer with no activation (relu added later after batch norm)
                layer = tf.layers.dense(inputs=layer,
                                    activation=tf.nn.tanh,
                                    units=n,
                                    name='DenseEmbedding'+str(idx))

                # avoid applying batchnorm to the last layer, otherwise the aggregation below
                # i.e. the reduce_mean will be always zero
                if idx < len(self.config['loss_embedding_layers']) - 1:
                    layer = tf.layers.batch_normalization(inputs=layer,
                                                          training=self.is_training,
                                                          name='BatchNorm' + str(idx))

            # the aggregation layer takes the mean of all the embedded loss terms
            # for every inputed pairs of estimated and true target values;
            # it outputs a (1 x -1) single row aggregation tensor
            layer = tf.reshape(tf.reduce_mean(layer, axis=0),
                               shape=(1, -1),
                               name='AggregationLayer')

            # the network that takes the aggregated vector and projects it into
            # a further latent embedding
            for idx, n in enumerate(self.config['loss_aggregation_layers']):
                # no batch norm added here as the aggregation is a single row tensor
                layer = tf.layers.dense(inputs=layer,
                                    activation=tf.nn.tanh,
                                    units=n,
                                    name='DenseAggregation'+str(idx))

            # finally the estimated/surrogate loss converted to a scalar value
            self.L_hat = tf.reshape(tf.layers.dense(inputs=layer,
                                         activation=tf.nn.leaky_relu,
                                         units=1),
                                    shape=(),
                                    name='L_hat')

            # store the variables of the loss network
            self.loss_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='LossNetwork')

    # create the update rules for the first-order optimization
    def create_update_rules(self):
        with tf.variable_scope('UpdateRules'):

            # minimize the estimated loss wrt to the prediction network parameters
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PredictionNetwork')

            # clip the gradients of the prediction network and then apply them to
            # minimize the surrogate loss
            with tf.control_dependencies(update_ops):

                # the unclipped and clipped gradients of the surrogate wrt the params of the prediction model
                unclipped_grads = tf.gradients(self.L_hat, self.pred_variables)
                clipped_grads, _ = tf.clip_by_global_norm(unclipped_grads, self.config['max_grad_norm'])

                # the gradient norms of the surrogate loss wrt prediction model params
                self.pred_unclipped_grad_norm = tf.global_norm(unclipped_grads)
                self.pred_clipped_grad_norm = tf.global_norm(clipped_grads)

                # create an optimization update op for training the params of the prediction model
                self.prediction_model_update = tf.train.AdamOptimizer(self.config["eta_pred"]).\
                    apply_gradients(zip(clipped_grads, self.pred_variables))

            # minimize the difference between the true loss and the estimated loss
            # wrt to the parameters of the loss network
            self.loss_loss = tf.losses.absolute_difference(labels=self.L_true,
                                                           predictions=self.L_hat)

            # clip the gradients of the loss network and then apply them in order to minimize
            # the difference between the surrogate loss and the true loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='LossNetwork')

            with tf.control_dependencies(update_ops):

                # unclipped and clipped gradients of the reconstruction loss wrt the loss model params
                unclipped_grads = tf.gradients(self.loss_loss, self.loss_variables)
                clipped_grads, _ = tf.clip_by_global_norm(unclipped_grads, self.config['max_grad_norm'])

                # the gradient norms of the loss aproximation L1 term wrt loss params
                self.loss_unclipped_grad_norm = tf.global_norm(unclipped_grads)
                self.loss_clipped_grad_norm = tf.global_norm(clipped_grads)

                # an update op for training the params of the loss model
                self.loss_model_update = tf.train.AdamOptimizer(self.config["eta_loss"]). \
                    apply_gradients(zip(clipped_grads, self.loss_variables))

    # update the prediction model, return surrogate loss, the unclipped and clipped gradient norms
    def update_prediction_model(self, sess, X_batch, Y_batch):

            # update the prediction model parameters
            _, loss, grad_norm_unclipped, grad_norm_clipped = \
                sess.run(fetches=[self.prediction_model_update, self.L_hat,
                                  self.pred_unclipped_grad_norm, self.pred_clipped_grad_norm],
                         feed_dict={self.X: X_batch,
                                    self.Y_true: Y_batch,
                                    self.is_training: True})

            return loss, grad_norm_unclipped, grad_norm_clipped

    # update the loss model, return loss reconstruction, the unclipped and clipped gradient norms
    def update_loss_model(self, sess, X_batch, Y_batch, L_batch_true):

            # update the loss model parameters
            _, loss_performance, grad_norm_unclipped, grad_norm_clipped = \
                sess.run(fetches=[self.loss_model_update, self.loss_loss,
                                  self.loss_unclipped_grad_norm, self.loss_clipped_grad_norm],
                         feed_dict={self.X: X_batch,
                                    self.Y_true: Y_batch,
                                    self.L_true: L_batch_true,
                                    self.is_training: True})

            return loss_performance, grad_norm_unclipped, grad_norm_clipped
