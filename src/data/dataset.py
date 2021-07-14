import os

import numpy as np
import cv2
import tqdm
import torch
from torch.utils import data


class ThousandLandmarksDataset(data.Dataset):
    """Describes dataset format. Data should be put in folder
       with csv file for train/valdidation purpose r without it
       for prediction. Example of dataset could be find in root folder.
       
       :args:
            - root - folder with dataset.
            - transforms - list of transforms for dataset.
            - split - could be train, val or test, default is train.
            - train_size - size of train partition of dataset, default is 0.8
        :returns:
            - torch.data.Dataset object."""
       
    def __init__(self, root: str, transforms: list, split="train", train_size=0.8):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp), total=num_lines + 1):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(train_size * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(train_size * num_lines):
                    continue  # has not reached start of val part of data
                elements = line.strip().split(",")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)