from hyperparams import Hyperparams
from surrogate_model import SurrogateModel
from universal_loss.universal_loss_optimizers import UniversalLossOptimizer
from binarydataset import BinaryDataset
import sys

# get the default hyper parameters
hp = Hyperparams()
config = hp.get_default()
# set any other command line configurations in key=val format
for arg in sys.argv[1:]:
    key, raw_value = arg.split('=')
    # set binary values
    if key == 'stratified_batch' or key == 'init_universal_loss':
        value = raw_value.lower() == 'true'
        print('Binary configuration:', key, '=', value)
    # set float configurations
    elif key == 'eta_pred' or key == 'eta_loss' or key == 'max_grad_norm':
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

print('Configuration:', config)

# create the model
model = SurrogateModel(config=config)
model.create_model()

# create the optimizer
optimizer = UniversalLossOptimizer(config=config,
                                   model=model)
optimizer.run()
