import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from process_images import convert_image

class EEGImageNetDataset(Dataset):
    def __init__(self, args, transform=None):
        self.dataset_dir = args.dataset_dir
        self.transform = transform
        loaded = torch.load(os.path.join(args.dataset_dir, "EEG-ImageNet.pth"))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False

        # Store invalid indices for removal
        self.invalid_indices = []

    def __getitem__(self, index):
        if self.use_image_label:
            path = self.data[index]["image"]
            image_file = os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path)
            
            # print("-----")
            # print("imagefile: ", image_file)
            # print("path: ", path)
            # print("index: ", index)

            if not os.path.exists(image_file):
                # print(f"{image_file} DOES NOT EXIST, MARKING FOR REMOVAL")
                if not convert_image(path):
                    self.invalid_indices.append(index)
                    return None, path  # Returning None for invalid images

            label = Image.open(image_file)
            if label.mode == 'L':
                label = label.convert('RGB')
            if self.transform:
                label = self.transform(label)
            else:
                label = path
        else:
            label = self.labels.index(self.data[index]["label"])

        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]
        
        return feat, label

    def __len__(self):
        return len(self.data)

    def cleanup_invalid_data(self):
        """Remove all invalid indices from the dataset."""
        if self.invalid_indices:
            print(f"Removing {len(self.invalid_indices)} invalid entries from dataset.")
            self.data = [self.data[i] for i in range(len(self.data)) if i not in self.invalid_indices]
            self.invalid_indices = []  # Clear after cleanup

    def add_frequency_feat(self, feat):
        if len(feat) == len(self.data):
            self.frequency_feat = torch.from_numpy(feat).float()
        else:
            raise ValueError("Frequency features must have same length")
