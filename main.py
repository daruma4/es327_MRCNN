import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize


# Current working directory of the project
ROOT_DIR = os.path.abspath("")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Path to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Path to dataset
DATASET_DIR = "raws"

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
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + vessel(s)
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    DETECTION_MIN_CONFIDENCE = 0.8
    #Hyper Parameters
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

class VesselInferenceConfig(VesselConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_RESIZE_MODE = "pad64"
    DETECTION_MIN_CONFIDENCE = 0.8 # Detection threshold
############################################################
#  Dataset
############################################################

class VesselDataset(utils.Dataset):
    """_summary_

    Args:
        utils (_type_): _description_
    """
    def load_dataset(self, dataset_dir, is_train: bool):
        """Load training and validation images
        """
        # Add class(es)
        self.add_class("vessels", 1, "vessel")

        #Get image IDs from directory names
        image_ids = next(os.walk(os.path.join(ROOT_DIR, dataset_dir)))[2]

        #Add images
        for idx, image_id in enumerate(image_ids):
            if is_train and idx >= int(len(image_ids)*0.75):
                continue
            if not is_train and idx < int(len(image_ids)*0.75):
                continue
            mask_dir = os.path.join(ROOT_DIR, "masks")
            mask_file_name = "m" + image_id[1:]
            self.add_image(
                source="vessels",
                image_id=image_id,
                mask_path=os.path.join(mask_dir, mask_file_name),
                path=os.path.join(dataset_dir, image_id)
            )
    
    def load_mask(self, image_id: int):
        """Returns mask(s) and class_id for each mask.
        """
        #Read mask
        mask = cv2.imread(self.image_info[image_id]["mask_path"]).astype(np.int32)
        #Class ID array
        class_id_array = np.ones([mask.shape[-1]], dtype=np.int32)

        #Return mask and the corresponding class ID array
        return mask.astype(bool), class_id_array

    def image_reference(self, image_id):
        """Return path of the image.
        """
        info = self.image_info[image_id]
        return info["path"]
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = cv2.imread(self.image_info[image_id]["path"]).astype(np.int32)
        return image
############################################################
#  Train
############################################################

def train():
    config = VesselConfig()

    #Training dataset
    dataset_train = VesselDataset()
    dataset_train.load_dataset(DATASET_DIR, is_train=True)
    dataset_train.prepare()
    #Validation dataset
    dataset_val = VesselDataset()
    dataset_val.load_dataset(DATASET_DIR, is_train=False)
    dataset_val.prepare()

    #Create MaskRCNN model object
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    #Load COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    #Train
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads') #Train only head layers
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=90, layers='all') #Train all layers

    #Save
    model_save_path = os.path.join(DEFAULT_LOGS_DIR, f"mask_rcnn_lr_{config.LEARNING_RATE}_spe_{config.STEPS_PER_EPOCH}_vs_{config.VALIDATION_STEPS}_dmc_{config.DETECTION_MIN_CONFIDENCE}.h5")
    model.keras_model.save_weights(model_save_path)

############################################################
#  Inference
############################################################

def infer():
    #Load dataset
    dataset_val = VesselDataset()
    dataset_val.load_dataset(DATASET_DIR, is_train=False)
    dataset_val.prepare()

    #Generate inference config
    config = VesselInferenceConfig()
    #Create inference model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    #Load trained weights
    weights_path = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_quick.h5")
    model.load_weights(weights_path, by_name=True)

    class_names = ["BG", "vessel"]
    image_id = random.choice(dataset_val.image_ids)
    image = dataset_val.load_image(image_id=image_id)
    r = model.detect([image])[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])