import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import preprocess as preprocess  # Assuming this module handles preprocessing

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)  # Total number of samples

    def __getitem__(self, idx):
        while True:
            img_id = self.data_frame.loc[idx, 'image_id']
            img_path = os.path.join(self.root_dir, f"{img_id}.dcm")

            # Check if the image file exists
            if not os.path.exists(img_path):
                idx = (idx + 1) % len(self.data_frame)  # Move to the next index
                continue  # Try to fetch the next image

            try:
                # Preprocess the image
                processed_image = preprocess.remove_watermark(img_path)  # Ensure this returns a NumPy array
                processed_image = Image.fromarray(processed_image)  # Convert to PIL Image
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")
                idx = (idx + 1) % len(self.data_frame)  # Move to the next index
                continue  # Try to fetch the next image

            try:
                # Get the label
                label = int(self.data_frame.loc[idx, 'cancer'])
            except Exception as e:
                print(f"Error retrieving label for index {idx}: {e}")
                idx = (idx + 1) % len(self.data_frame)  # Move to the next index
                continue  # Try to fetch the next image

            try:
                # Apply any additional transformations
                if self.transform:
                    processed_image = self.transform(processed_image)
            except Exception as e:
                print(f"Error applying transformation: {e}")
                idx = (idx + 1) % len(self.data_frame)  # Move to the next index
                continue  # Try to fetch the next image

            return processed_image, label

