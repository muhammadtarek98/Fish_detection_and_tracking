import torch
import torchvision
import cv2
import os
import numpy as np
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, mask_dir: str,
                 transform: torchvision.transforms = None) -> None:
        super(CustomDataset,self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(sorted(os.listdir(image_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))
        assert len(self.images) == len(self.masks), "Number of images and masks should match."
    def __len__(self) -> int:
        return len(self.images)

    def image_enhancement(self,image:np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blue_channel, green_channel, red_channel = cv2.split(image)
        equalized_blue_channel = clahe.apply(blue_channel)
        equalized_green_channel = clahe.apply(green_channel)
        equalized_red_channel = clahe.apply(red_channel)
        equalized_image = cv2.merge([equalized_blue_channel, equalized_green_channel, equalized_red_channel])
        return equalized_image

    def __getitem__(self, idx:int):
        image = cv2.imread(filename=os.path.join(self.image_dir, self.images[idx]))
        image=self.image_enhancement(image)
        mask =  cv2.imread(filename=os.path.join(self.mask_dir, self.masks[idx]))
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask=torch.from_numpy(mask)
        image = image.astype(np.float32)
        mask = mask.to(torch.float32)
        object_ids = torch.unique(mask)[1:]
        num_object_ids = len(object_ids)
        mask = (mask == object_ids[:, None, None]).to(dtype=torch.uint8)
        #the BB is (xmin, ymin, xmax, ymax)
        boxes = torchvision.ops.masks_to_boxes(masks=mask[0:])
        iscrowd=torch.zeros(num_object_ids,dtype=torch.int64)
        labels = torch.ones(num_object_ids, dtype=torch.int64)
        target = {
            "image_dir":os.path.join(self.image_dir, self.images[idx]),
            "image_idx": idx,
            'boxes': boxes,
            'labels': labels,
            "iscrowd": iscrowd,
        }
        if self.transform:
            augmented_example= self.transform(image=image,
                                              labels=target['labels'],
                                              bboxes=target['boxes'])
            image=augmented_example['image']
            target['labels']=torch.tensor(augmented_example['labels'])
            target['boxes']=torch.tensor(augmented_example['bboxes'])
        return image, target