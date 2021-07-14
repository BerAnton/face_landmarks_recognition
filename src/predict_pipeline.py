import os
import yaml
import pickle

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import LandmarkModel
from src.utils import restore_landmarks_batch, make_csv
from src.data import ThousandLandmarksDataset
from src.transforms import ScaleMinSideToSize, CropCenter, TransformByKeys


def predict(model, loader, device, num_pts):
    model.eval()
    predictions = np.zeros((len(loader.dataset), num_pts, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="Test prediction...")):
        images = batch["image"].to(device)
        
        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), num_pts, 2)) # B * NUM_PTS * 2
        
        fs = batch["scale_coef"].numpy() # B
        margins_x = batch["crop_margin_x"].numpy() # B
        margins_y = batch["crop_margin_y"].numpy() # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y) # B * NUM_PTS * 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction
        
    return predictions

def predict_pipeline(predict_config_path):
    """Train pipeline for landmarks recognition.
       :args:
            - predict_config_path - path to config with predict params."""
    
    with open(predict_config_path, "r") as fin:
        predict_config = yaml.safe_load(fin)

    input_data_path = predict_config["input_data_path"]
    output_path = os.path.join(predict_config["output_data_path"], predict_config["output_filename"])
    model_path = os.path.join(predict_config["model_path"], predict_config["model_name"])
    batch_size = predict_config["batch_size"]
    crop_size = predict_config["crop_size"]
    num_pts = predict_config["num_pts"]
    cuda = predict_config["use_cuda"]
    csv_file_name = predict_config["csv_filename"]

    test_transforms = transforms.Compose([
        ScaleMinSideToSize((crop_size, crop_size)),
        CropCenter(crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",))
    ])

    device = torch.device("cuda") if cuda else torch.device("cpu")

    print("Data loadnig")
    test_dataset = ThousandLandmarksDataset(input_data_path, test_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    model = LandmarkModel()
    with open(model_path, "rb") as fin:
        best_state_dict = torch.load(fin, map_location=device)
        model.load_state_dict(best_state_dict)
    
    test_predictions = predict(model, test_dataloader, device, num_pts)

    with open(output_path, "wb") as fout:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fout)

    make_csv(input_data_path, output_path, csv_file_name)