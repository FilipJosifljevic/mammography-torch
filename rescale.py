import numpy as np
from PIL import Image

rescale_factor = 255/65535

class RescaleTransform:
    def __call__(self, image):
        image_array = np.array(image, dtype=np.float32)
        image_array *= rescale_factor
        image_array = np.clip(image_array, 0, 255)
        
        return Image.fromarray(image_array.astype(np.uint8))