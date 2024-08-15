import torch
import torch_npu
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import wandb
from custom_dataset import CustomDataset
from rescale import RescaleTransform
from torchvision import datasets
from torch_mammo import CustomCNN
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if run on CUDA use this
print(torch.npu.is_available())  # True
print(torch.npu.device_count())  # 2
print(torch.npu.current_device())
device = torch.device("npu:0")
torch.npu.set_device(device) #this and the row before if run on an NPU

wandb.init(project="mammo-torch", config={"batch_size": 8, "learning_rate": 0.001, "epochs":10, "workers":32})
config = wandb.config


transform = transforms.Compose([
    RescaleTransform(),
    transforms.Resize((1152, 896)),  # Resize images to a fixed size
    transforms.Grayscale(num_output_channels=3),  # Ensure grayscale images are 3-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3-channel images
])

train_dataset = CustomDataset(csv_file='./train.csv', root_dir='./train_img', transform=transform)
#valid_dataset = CustomDataset(csv_file='./train.csv', root_dir='./valid_img', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

model = CustomCNN().to(device)
model_path = './inbreast_vgg16_512x1.pth'
model.load_state_dict(torch.load(model_path))
model.freeze_except_last_two = True

wandb.watch(model, log_freq=100)

optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)


def calculate_metrics(predictions, labels):

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    return precision, recall, f1

print('===========>Starting Training<===========')

for epoch in range(config.epochs):
    
    model.train()
    running_loss =0.0
    all_labels = []
    all_predictions = []

    
    for i, data in enumerate(train_loader, 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        running_loss += loss.item()
        if i % 2000 == 1999:
            avg_loss = running_loss / 2000
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
            running_loss = 0.0

    precision, recall, f1 = calculate_metrics(torch.tensor(all_predictions), torch.tensor(all_labels))
    wandb.log({
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
        "epoch": epoch + 1
    })

print('===========>Finished Training<===========')
print('Saving Model.....')
torch.save(model.state_dict(), './inbreast_vgg16_512x1.pth')
print('Model Saved!')
