import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from torch_mammo import CustomCNN
from sklearn.metrics import confusion_matrix

torch_model = CustomCNN()
torch_model.load_state_dict(torch.load('./inbreast_vgg16_512x1.pth'))

torch_model.eval()

test_csv_file = pd.read_csv('./test.csv')
#drop the rows that have nan in cancer column
test_csv_file.dropna(subset=['cancer'], inplace=True)

true_labels = []
predicted_labels = []

for index, row in test_csv_file.iterrows():

    image_id = row['image_id']
    true_label = row['cancer']
    image_path = f'./test_img/{image_id}.png'
    #because some preprocessed images are inserted into the test folder in lack of the amount of images
    if not os.path.exists(image_path):
        image_path = f'./test_img/{image_id}.dcm_preprocessed.png'

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (896, 1152), cv2.INTER_CUBIC)
    img = np.stack([img] * 3)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    with torch.no_grad():
        pred = torch_model(img)
        probabilities = F.softmax(pred, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    true_labels.append(true_label)
    predicted_labels.append(prediction)

    print(f'Image ID: {image_id} | True Label: {true_label} | Prediction: {prediction}')


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

print('Confusion Matrix : ')
print(conf_matrix)
