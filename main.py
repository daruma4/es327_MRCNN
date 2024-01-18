import os
import cv2
import numpy as np

# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize


# Current working directory of the project
ROOT_DIR = os.getcwd()
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

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
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 1 # background + vessel(s)
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    #Hyper Parameters
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5

############################################################
#  Dataset
############################################################

class VesselDataset(utils.Dataset):
    """Imports US dataset

    Args:
        utils (_type_): _description_
    """
    def load_images(self, dataset_dir):
        """_summary_

        Args:
            dataset_dir (os.path): e.g. "raws"
        """
        # Add class(es)
        self.add_class("vessels", 1, "vessel")

        #Get image IDs from directory names
        image_ids = next(os.walk(os.path.join(os.getcwd(), dataset_dir)))[2]

        #Add images
        for image_id in image_ids:
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
        mask = cv2.imread(mask_path)
        #Class ID array
        class_id_array = np.ones([mask.shape[-1]], dtype=np.int32)

        #Return mask and the corresponding class ID array
        return mask, class_id_array

config = VesselConfig()
dataset = VesselDataset()
dataset.load_images("raws")
dataset.prepare()
# mask, class_ids = dataset.load_mask(0)
# visualize.display_top_masks(dataset.load_image(0), mask, class_ids, dataset.class_names, limit=1)
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, config, 0, use_mini_mask=False)
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names,
#                             show_bbox=False)