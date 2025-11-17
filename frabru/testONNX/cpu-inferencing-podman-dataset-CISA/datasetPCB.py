from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from modelConfig import ModelConfig
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import numpy as np


class PCBDataset(Dataset):

    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(fraction=0.7):
    data_list = []
    data_list_test = []
    class_to_idx = {cls: idx for idx, cls in enumerate(ModelConfig.classes)}

    

    print("Loading Dataset...")
    for pcb_dir in ModelConfig.pcb_dirs:
        pcb_path = os.path.join(ModelConfig.data_root, pcb_dir)
    
        csv_file = list(Path(pcb_path).glob("*.csv"))

        print(f"File csv_file: {csv_file[0].absolute()}")

        if csv_file:
            df = pd.read_csv(csv_file[0].absolute())

            df_subset_training, df_subset_test = train_test_split(
                df, 
                train_size=ModelConfig.train_perc,
                random_state=42
            )

            for (_, row) in df_subset_training.iterrows():
                img_path = row['image']

                cls = row['label']
                
                
                if cls and cls in class_to_idx.keys():
                    img_path = Path(ModelConfig.data_root) / img_path
                    data_list.append((img_path, class_to_idx[cls]))

            for (_, row_test) in df_subset_test.iterrows():
                img_path_test = row_test['image']
                cls_test = row_test['label']

                if cls_test and cls_test in class_to_idx.keys():
                    img_path_test = Path(ModelConfig.data_root) / img_path_test
                    data_list_test.append((img_path_test, class_to_idx[cls_test]))


    print(f"Total image loaded for training: {len(data_list)}")
    print(f"Total image loaded for test: {len(data_list_test)}")


    class_counts = {"Training": {}, "Test": {}}

    for (_,label) in data_list:
        cls_name = ModelConfig.classes[label]
        class_counts["Training"][cls_name] = class_counts["Training"].get(cls_name, 0) + 1

    for (_, label_test) in data_list_test:
        cls_name_test = ModelConfig.classes[label_test]
        class_counts["Test"][cls_name_test] = class_counts["Test"].get(cls_name_test, 0) + 1


    print("\nClass Distrib.:")
    for phase, dictClasses  in class_counts.items():
        print(f"{phase}")
        for cls, count in dictClasses.items():
            print(f"  {cls}: {count}")

    return data_list, data_list_test



train_transform = transforms.Compose([
    transforms.Resize((ModelConfig.img_size, ModelConfig.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((ModelConfig.img_size, ModelConfig.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((ModelConfig.img_size, ModelConfig.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
