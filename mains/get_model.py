from models.VGG_model import VGGModel
from models.Resnet_model import ResnetModel
from models.Inception_model import InceptionModel
def model(model_name,config):
    if model_name=="VGG":
        return VGGModel(config)
    elif model_name=="RESNET":
        return ResnetModel(config)
    elif model_name=="INCEPTION":
        return InceptionModel(config)
    else:
        raise Exception("Model name is wrong!")