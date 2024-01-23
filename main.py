import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize


# Current working directory of the project
ROOT_DIR = os.getcwd()
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Path to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class VesselConfig(Config):
    """Configuration for training on the US dataset. Derives from the base Config class and overrides values specific to the US dataset.

    Args:
        Config (Config): default Config class from mrcnn.config
    """

    NAME = "vessels"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1 # background + vessel(s)
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    #Hyper Parameters
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

############################################################
#  Dataset
############################################################

class VesselDataset(utils.Dataset):
    """Imports US dataset

    Args:
        utils (_type_): _description_
    """
    def load_images(self, dataset_dir, is_train: bool):
        """_summary_

        Args:
            dataset_dir (os.path): e.g. "raws"
        """
        # Add class(es)
        self.add_class("vessels", 1, "vessel")

        #Get image IDs from directory names
        image_ids = next(os.walk(os.path.join(os.getcwd(), dataset_dir)))[2]

        #Add images
        for idx, image_id in enumerate(image_ids):
            if is_train and idx >= int(len(image_ids)*0.75):
                continue
            if not is_train and idx < int(len(image_ids)*0.75):
                continue
            self.add_image(
                "vessels",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id)
            )
    
    def load_mask(self, image_id: int):
        """Returns the mask of a specific image

        Args:
            image_id (int): Image ID

        Returns:
            mask: _description_
            class_id_array: _description_
        """
        #Get image info
        info = self.image_info[image_id]
        #Setup mask directory
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info["path"])), "masks")
        mask_name = "m" + os.path.basename(info["path"])[1:]
        mask_path = os.path.join(mask_dir, mask_name)
        #Read mask
        mask = cv2.imread(mask_path)[:,:,:].astype(np.uint8)
        #Class ID array
        class_id_array = np.ones([mask.shape[-1]], dtype=np.int32)

        #Return mask and the corresponding class ID array
        return mask.astype(bool), class_id_array

############################################################
#  Run
############################################################

config = VesselConfig()
dataset_train = VesselDataset()
dataset_train.load_images("raws", is_train=True)
dataset_train.prepare()
dataset_val = VesselDataset()
dataset_val.load_images("raws", is_train=False)
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR) #Create MaskRCNN model object
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"]) #Load COCO weights

#Train
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads') #Train only head layers
# model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=90, layers='all') #Train all layers