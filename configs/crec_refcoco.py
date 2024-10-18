from crec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.crec import model


# Choose from ['refcoco', 'refcoco+', 'refcocog', 'c-refcoco', 'c-refcoco+', 'c-refcocog']
dataset.dataset = 'refcoco'

# Image/Annotation path
dataset.ann_path["refcoco"] = "./data/rec/refcoco.json"
dataset.image_path["refcoco"] = "./data/images/train2014"

dataset.ann_path["refcoco+"] = "./data/rec/refcoco+.json"
dataset.image_path["refcoco+"] = "./data/images/train2014"

dataset.ann_path["refcocog"] = "./data/rec/refcocog.json"
dataset.image_path["refcocog"] = "./data/images/train2014"


dataset.ann_path["c-refcoco"] = "./data/crec/c_refcoco.json"
dataset.image_path["c-refcoco"] = "./data/images/train2014"

dataset.ann_path["c-refcoco+"] = "./data/crec/c_refcoco+.json"
dataset.image_path["c-refcoco+"] = "./data/images/train2014"

dataset.ann_path["c-refcocog"] = "./data/crec/c_refcocog.json"
dataset.image_path["c-refcocog"] = "./data/images/train2014"


# Vocabulary path 
## if do not exist, it will build new vocabulary and pretrained_emb to the paths
## Better to build different vocabulary files for different datasets
dataset.ann_path["vocabulary"] = "./output/vocabulary.json"
dataset.ann_path["pretrained_emb"] = "./output/pretrained_emb.npy"

# Training cfg
train.output_dir = "./output/refcoco"
train.auto_resume.enabled = False
train.epochs = 50
train.batch_size = 32
train.save_period = 5
train.log_period = 100
train.evaluation.eval_batch_size = 64
train.sync_bn.enabled = False

# Optim
train.scheduler.name = "step"
train.base_lr = 1e-4
optim.lr = train.base_lr

# Model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path="./data/weights/cspdarknet_coco.pth"
