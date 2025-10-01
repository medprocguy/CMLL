import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import os
import numpy as np


class COCOMultiLabelDataset(Dataset):
    def __init__(self, root, annFile, ids, transform=None, num_classes=80):
        """
        Args:
            root (str): Path to COCO images folder (train2017/val2017).
            annFile (str): Path to COCO annotation file (instances_train2017.json).
            transform: Torchvision transforms for images.
            num_classes (int): Number of COCO categories (default 80).
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = ids
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # load image
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        # load annotations (multi-label)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        labels = [ann['category_id'] for ann in anns]

        # multi-hot encode labels
        target = np.zeros(self.num_classes, dtype=np.float32)
        for l in labels:
            # COCO category IDs are not 0..79, remap them!
            cat_idx = self.coco.getCatIds(catIds=[l])[0] - 1
            if 0 <= cat_idx < self.num_classes:
                target[cat_idx] = 1.0
        
        return img, torch.tensor(target)

def build(args, train_transforms=None, eval_transforms=None):
    image_root = '../COCO/train2014'
    annFile="../COCO/annotations/instances_train2014.json"
     
    all_ids = list(COCO(annFile).imgs.keys())
    random.shuffle(all_ids)
    l = int(0.8*len(all_ids))
    train_dataset = COCOMultiLabelDataset(root=image_root, annFile=annFile, ids=all_ids[:l], transform=train_transforms, num_classes=80)
    val_dataset = COCOMultiLabelDataset(root=image_root, annFile=annFile, ids=all_ids[l:], transform=eval_transforms, num_classes=80)

    return train_dataset, val_dataset
    
def build_test(args, transforms=None):
    image_root = '../COCO/val2014'
    annFile="../COCO/annotations/instances_val2014.json"
     
    all_ids = list(COCO(annFile).imgs.keys())
    test_dataset = COCOMultiLabelDataset(root=image_root, annFile=annFile, ids=all_ids, transform=transforms, num_classes=80)
    
    return test_dataset    

