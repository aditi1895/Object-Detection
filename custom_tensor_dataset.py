from torch.utils.data import Dataset
import numpy as np

class CustomTensorDataset(Dataset):

    def __init__(self, tensors, transforms = None) -> None:
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
    
        image = image.transpose(2,0,1)

        # if self.transforms:
        #     image = self.transforms(image)
        
        return(image, label, bbox)

    def __len__ (self):

        return len(self.tensors[0])