#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
from PIL import Image
import os

def get_split_percent_as_str(train_df, valid_df, test_df):
  total_size = train_df.size + valid_df.size + test_df.size
  train_ratio = train_df.size/total_size*100
  valid_ratio = valid_df.size/total_size*100
  test_ratio = test_df.size/total_size*100
  return f"train({train_ratio:.3})_valid({valid_ratio:.3})_test({test_ratio:.3})"

def prepare_split_dataset(dataset_path, split_path):
  data_df = pd.read_csv(dataset_path, index_col=0).reset_index()
  split_df = pd.read_csv(split_path, index_col=0).reset_index()
  data_df = pd.merge(data_df, split_df, on="index")
  data_df = data_df[~data_df.split.isna()].drop("index", axis=1)

  train_df = data_df[data_df.split=="train"].loc[:, data_df.columns != "split"]
  validation_df = data_df[data_df.split=="validate"].loc[:, data_df.columns != "split"]
  test_df = data_df[data_df.split=="test"].loc[:, data_df.columns != "split"]

  return train_df, validation_df, test_df

class Chexpert(Dataset):
    def __init__(self, dataset_csv, split_csv, split, root="", transform=None):
        train, validation, test = prepare_split_dataset(dataset_csv, split_csv)
        if split == "train" :
            self.dataset  = train
        elif split == "validation" :
            self.dataset = validation
        elif split  == "test" :
            self.dataset = test
        else:
            raise Exception("Split should be in ['train', 'test', 'valid'")
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def get_pathology_labels(self):
        return self.dataset.columns[1:15].values.tolist()

    def get_race_labels(self):
        return self.dataset.columns[15:].values.tolist()

    def __getitem__(self, index):
        image_path = self.dataset.iloc[index][0]
        pathology_labels = self.dataset.iloc[index][1:15].values
        race_labels = self.dataset.iloc[index][15:].values
        image = Image.open(os.path.join(self.root, image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, pathology_labels, race_labels

class DummyChexpert(Dataset):
    def __init__(self, dataset_csv, split_csv, split, root="", transform=None):
        train, validation, test = prepare_split_dataset(dataset_csv, split_csv)
        if split == "train" :
            self.dataset  = train
        elif split == "validation" :
            self.dataset = validation
        elif split  == "test" :
            self.dataset = test
        else:
            raise Exception("Split should be in ['train', 'test', 'validation'")
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def get_pathology_labels(self):
        return self.dataset.columns[1:15].values.tolist()

    def get_race_labels(self):
        return self.dataset.columns[15:].values.tolist()

    def __getitem__(self, index):
        image_path = self.dataset.iloc[index][0]
        pathology_labels = self.dataset.iloc[index][1:15].values.astype(np.int32)
        race_labels = self.dataset.iloc[index][15:].values.astype(np.int32)
        image = Image.fromarray(np.random.rand(224,224,3), 'RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, pathology_labels, race_labels


class TwoHeadModel(nn.Module):
    def __init__(self, features, head1_size, head2_size):
        super(TwoHeadModel, self).__init__()
        self.features = features
        self.head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, head1_size)
        )
        self.head2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, head2_size)
        )

    def forward(self, x):
        x = self.features(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2

MODEL = "densenet121"
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
ROOT = "./"
N_EPOCHS = 20
#%%
model = models.__dict__[MODEL]()
myModel = TwoHeadModel(model.features, 14, 3)
device = torch.device("cpu")
myModel.to(device)

my_transforms = transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.1),
    transforms.ToTensor(),
    NORMALIZE
])

valid_dataset = DummyChexpert("../Datasets/Chexpert/csv/pathology_race_train.csv", 
                                            "../Datasets/Chexpert/Splits/chexpert_split_0.672_0.12_0.208.csv",
                                            "validation", ROOT,
                                            transform=my_transforms)
validation_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(myModel.parameters(), lr=0.01)

pathology_loss = nn.BCELoss()
race_loss = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir="./")

for epoch in range(1, N_EPOCHS):
    losses = []
    race_accuracies = []
    all_pathology_scores = []
    all_pathology_targets = []

    for i, (images, pathology, race) in enumerate(validation_dataloader):
        images = images.to(device)
        pathology = pathology.type(torch.float32).to(device)
        race = race.type(torch.float32).to(device)
        optimizer.zero_grad()

        pathology_logits, race_logits = myModel(images)
        pathology_scores = torch.sigmoid(pathology_logits)
        race_proba = F.softmax(race_logits, dim=1)
        
        #break 
        loss = pathology_loss(pathology, pathology_scores) + race_loss(race, race_proba)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        race_accuracies.append( (race.argmax(axis=1) == race_logits.argmax(axis=1)).sum()/len(race)*100)
        all_pathology_scores.extend(pathology_scores.cpu().detach().numpy().tolist())
        all_pathology_targets.extend(pathology.cpu().detach().numpy().tolist())
        #break
    writer.add_scalar("train_loss", np.mean(losses), global_step=epoch)
    writer.add_scalar("train_race_accuracy", np.mean(race_accuracies), global_step=epoch)
    writer.add_scalar("train_auc_roc", roc_auc_score(all_pathology_targets, all_pathology_scores))
    #break



    


# %%
