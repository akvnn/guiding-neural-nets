import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from typing import Literal
from torch.utils.data import Dataset
import math
from pathlib import Path


class CustomDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        base_dir: str,
        task: Literal["condition", "treatment", "both"],
    ):
        self.conditions = [
            "Cavitated",
            "Retained Root",
            "Crowned",
            "Filled",
            "Impacted",
            "Implant",
        ]
        self.treatments = ["Filling", "Root Canal", "Extraction", "None"]
        
        self.base_dir = Path(base_dir)
        self.annotations = pd.read_csv(
            self.base_dir / annotations_file, na_values=[None]
        )
        self.annotations = self.annotations.replace(np.nan, "None")
        self.task = task
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                self.min_max_normalize,
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        
    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def get_binary_mask(image, x, y, width, height, rotation):
        """Returns binary mask of the image with the rotated bounding box as the region of interest."""
        img_width = image.shape[1]
        img_height = image.shape[0]

        w, h = width * img_width / 100, height * img_height / 100
        a = math.pi * (rotation / 180.0) if rotation else 0.0
        cos_a, sin_a = math.cos(a), math.sin(a)

        x1, y1 = x * img_width / 100, y * img_height / 100  # top left
        x2, y2 = x1 + w * cos_a, y1 + w * sin_a  # top right
        x3, y3 = x2 - h * sin_a, y2 + h * cos_a  # bottom right
        x4, y4 = x1 - h * sin_a, y1 + h * cos_a  # bottom left

        coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([coords], dtype=np.int32), 255)
        return mask

    @staticmethod
    def get_sub_image(image, x, y, width, height, rotation):
        rotation *= -1
        original_height, original_width = image.shape[:2]
        pixel_x = x / 100.0 * original_width
        pixel_y = y / 100.0 * original_height
        pixel_width = width / 100.0 * original_width
        pixel_height = height / 100.0 * original_height
        center_x = pixel_x + pixel_width / 2
        center_y = pixel_y + pixel_height / 2

        rotation_matrix = np.array(
            [
                [np.cos(np.radians(rotation)), -np.sin(np.radians(rotation)), 0],
                [np.sin(np.radians(rotation)), np.cos(np.radians(rotation)), 0],
                [0, 0, 1],
            ]
        )

        translation_matrix_to_origin = np.array(
            [[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]]
        )
        translation_matrix_back = np.array(
            [[1, 0, pixel_width / 2], [0, 1, pixel_height / 2], [0, 0, 1]]
        )

        affine_matrix = np.dot(rotation_matrix, translation_matrix_to_origin)
        affine_matrix = np.dot(affine_matrix, translation_matrix_back)

        rotated = cv2.warpPerspective(
            image, affine_matrix, (int(pixel_width), int(pixel_height))
        )
        return rotated

    @staticmethod
    def get_normalized_teeth_coords(image, x, y, width, height, rotation):
        img_width = image.shape[1]
        img_height = image.shape[0]

        w, h = width * img_width / 100, height * img_height / 100
        a = math.pi * (rotation / 180.0) if rotation else 0.0
        cos_a, sin_a = math.cos(a), math.sin(a)

        x1, y1 = x * img_width / 100, y * img_height / 100  # top left
        x2, y2 = x1 + w * cos_a, y1 + w * sin_a  # top right
        x3, y3 = x2 - h * sin_a, y2 + h * cos_a  # bottom right
        x4, y4 = x1 - h * sin_a, y1 + h * cos_a  # bottom left

        return (
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height,
            x3 / img_width,
            y3 / img_height,
            x4 / img_width,
            y4 / img_height,
        )

    @staticmethod
    def get_cropped_image(image, x, y, width, height, zoom_factor=5):
        """Inputs x,y,width,height are in percentage (0-100)"""
        # Convert percentage values to pixel values
        img_height, img_width = image.shape[:2]
        x = int(x / 100.0 * img_width)
        y = int(y / 100.0 * img_height)
        width = int(width / 100.0 * img_width)
        height = int(height / 100.0 * img_height)
        # Calculate the bounding box around the target
        new_x = max(x - width // 2, 0)
        new_y = max(y - height // 2, 0)
        new_width = int(min(width * zoom_factor, img_width - new_x))
        new_height = int(min(height * zoom_factor, img_height - new_y))
        # Crop the image
        cropped_image = image[new_y : new_y + new_height, new_x : new_x + new_width]
        return cropped_image

    def process_row(self, row):
        img_path = row["image"]
        img_path = img_path + ".jpg" if "jpg" not in img_path else img_path
        label_condition = self.conditions.index(row["condition"])
        label_treatment = self.treatments.index(row["treatment"])
        folder: Path = self.base_dir / img_path.split("_")[0]

        img_path = folder / img_path
        grayscale_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        x, y, width, height, rotation = (
            row["x"],
            row["y"],
            row["width"],
            row["height"],
            row["rotation"],
        )
        sub_image = self.get_sub_image(grayscale_img, x, y, width, height, rotation)

        coords = torch.tensor(
            self.get_normalized_teeth_coords(
                grayscale_img, x, y, width, height, rotation
            ),
            dtype=torch.float32,
        )

        binary_mask = self.get_binary_mask(grayscale_img, x, y, width, height, rotation)

        grayscale_img = self.get_cropped_image(
            grayscale_img, x, y, width, height, zoom_factor=3
        )  # crop the image around the tooth (or prosthetic) with some zoom factor -> 2nd channel

        binary_mask = self.get_cropped_image(
            binary_mask, x, y, width, height, zoom_factor=3
        )

        sub_image = torch.from_numpy(sub_image).unsqueeze(0).float()
        sub_image = self.transforms(sub_image)

        grayscale_img = torch.from_numpy(grayscale_img).unsqueeze(0).float()
        grayscale_img = self.transforms(grayscale_img)

        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
        binary_mask = self.transforms(binary_mask)
        data = torch.cat([sub_image, grayscale_img, binary_mask], dim=0).to(
            dtype=torch.float32
        )

        return data, coords, label_condition, label_treatment

    @staticmethod
    def min_max_normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

    # ADDED THESE TWO FUNCTIONS
    def get_condition_labels(self):
        return self.annotations["condition"]

    def get_treatment_labels(self):
        return self.annotations["treatment"]

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        *data, label_condition, label_treatment = self.process_row(row)
        if self.task == "condition":
            return *data, label_condition
        elif self.task == "treatment":
            return *data, label_treatment
        else:
            return *data, label_condition, label_treatment
