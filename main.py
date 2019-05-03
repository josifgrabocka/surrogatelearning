from hyperparams import Hyperparams
from surrogate_model import SurrogateModel
from optimizer import Optimizer
from binarydataset import BinaryDataset
import sys

# get the default hyper parameters
hp = Hyperparams()
config = hp.get_default()
# set any other command line configurations in key=val format
hp.set_command_line_args(argv=sys.argv[2:], config=config)

print('Configuration:', config)

# read the dataset
ds = BinaryDataset(folder_path=sys.argv[1],
                   config=config)

# set the number of features, useful for the feature batch placeholder in the surogate model
config['num_features'] = ds.num_features

# create the surogate model
model = SurrogateModel(config=config)
model.create_model()

# create the optimizer
optimizer = Optimizer(config=config,
                      dataset=ds,
                      model=model)
optimizer.run()
