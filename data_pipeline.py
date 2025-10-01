import os 
import cv2 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 

# the blueprint from first principles 
class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # path to dir with normal and pneumonia
        self.data_dir = data_dir
        self.transform = transform 
        self.image_paths = []
        self.labels = []

        # find all image paths and assign labels - 9 for normal, 1 for pneumonia 
        for label, category in enumerate(['NORMAL', 'PNEUMONIA']):
            category_dir = os.path.join(self.data_dir, category)
            for image_name in os.listdir(category_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(category_dir, image_name))
                    self.labels.append(label)
        
        print(f"Found {len(self.image_paths)} images in {self.data_dir}")

    def __len__(self):
        # this tells dataloader how many total items are in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # this tells dataloader how to get a single item - image + label
        
        # load image with opencv
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        # convert from bgr to rgb 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get the corresponding label
        label = self.labels[idx]
        # apply transformations - resizing, augmenting, converting to tensor 
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# define transformations = for the training set, we apply data augmentation to make the model more robust 
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# for the valudation/test set, we only resize, convert and normalize 

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# create the datasets and data-loaders
base_dir = 'chest_xrays/chest_xray'
train_dataset = XRayDataset(data_dir = os.path.join(base_dir, 'train'), transform=train_transform)
val_dataset = XRayDataset(data_dir = os.path.join(base_dir, 'val'), transform=val_transform)

# the factory manager that creates batches of data
train_loader = DataLoader(dataset = train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = 16, shuffle = False)

# confirm pipeline is working by pullng one batch 
if __name__ == '__main__':
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # get one batch of images and labels 
    images, labels = next(iter(train_loader))

    #print the shape of the batch 
    print(f"\nShape of one batch of images: {images.shape}") # batch_size, channels, height, width
    print(f"Shape of oen batch of labels: {labels.shape}") # [batch_size]




















    