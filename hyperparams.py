import pickle

class Hyperparams:

    def __init__(self):
        pass

    # the default hyper-parameters
    def get_default(self):

        # init a dictionary of hyper-parameters
        config = {}

        # the layer dimensions
        config["prediction_layers"] = [100, 30, 10]
        # the capacity of the loss network, the embedding and aggregation layers
        config["loss_embedding_layers"] = [30, 30]
        config["loss_aggregation_layers"] = [10, 10]

        # the learning rates for updating the prediction model and loss model
        config["eta_pred"] = 0.00001
        config["eta_loss"] = 0.00001

        # use gradient clipping to avoid exploding gradients
        config["max_grad_norm"] = 0.00001

        # whether the sampling of batches should be stratified, or entirely random
        config["stratified_batch"] = True
        # if stratified, then what is the proportion of positive instances in one batch
        config["stratification_pos_ratio"] = 0.5
        # the batch size
        config["batch_size"] = 100

        # the loss type to be optimized
        config["loss"] = 'auc'
        # the dropout rate for regularization
        config['dropout_rate'] = 0.2

        # overall the prediction model is trained for config["num_epochs"]*config["steps_prediction"] mini-batches
        # similarily the loss model is trained for config["num_epochs"]*config["steps_loss"] mini-batches

        # the number of epochs to perform the regularization
        config["num_epochs"] = 300001
        # the number of mini-batch update steps for learning the loss model parameters
        config["steps_loss"] = 10
        # the number of mini-batch update steps for learning
        config["steps_prediction"] = 3


        # whether to initialize the loss with a universal one
        config["init_universal_loss"] = True
        # the folder location of the saved universal loss models
        config["universal_loss_folder"] = 'saved_loss_models/'

        # after how many batches should the intermediate results be plotted to the standard output
        # and the checkpoint of the model be saved
        config["performance_epoch_frequency"] = 1000

        # default number of features setting, needs to be overriden after reading
        # a dataset
        config["num_features"] = 1

        # some joker float parameters, that could be anything
        config["alpha"] = 1.0
        config["beta"] = 1.0
        config["gamma"] = 1.0


        return config

    # set command line arguments into an existing configuration
    def set_command_line_args(self, argv, config):

        for arg in argv:
            key, raw_value = arg.split('=')
            # set binary values
            if key == 'stratified_batch' or key == 'init_universal_loss':
                value = raw_value.lower() == 'true'
                print('Binary configuration:', key, '=', value)
            # set float configurations
            elif key == 'eta_pred' or key == 'eta_loss' or key == 'max_grad_norm' \
                    or key == 'stratification_pos_ratio' or key == 'dropout_rate' \
                    or key == 'alpha' or key == 'beta' or key == 'gamma':
                value = float(raw_value)
                print('Float configuration:', key, '=', value)
            # set int configurations
            elif key == 'batch_size' or key == 'num_epochs' or key == 'steps_loss' \
                    or key == 'steps_prediction' or key == 'performance_epoch_frequency':
                value = int(raw_value)
                print('Int configuration:', key, '=', value)
            # set list configurations
            elif key == 'prediction_layers' or key == 'loss_embedding_layers' \
                    or key == 'loss_aggregation_layers':
                value = list(map(int, raw_value.split(',')))
                print('List configuration:', key, '=', value)
            # set string configurations
            else:
                value = raw_value
                print('String configuration:', key, '=', value)

            # set the configuration
            config[key] = value


    def save(self, file_name):
        f = open(file_name, "wb")
        pickle.dump(dict, f)
        f.close()

    def load(self, file_name):
        f = open(file_name, "wb")
        pickle.dump(dict, f)
        f.close()
