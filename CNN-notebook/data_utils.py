
import torch
from torch.utils.data import Dataset, DataLoader, random_split

batch_size = 64
num_workers = 4

class CarDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # if self.transform:
        #    if torch.is_tensor(image):
        #       image = image.squeeze().numpy()
        #    transformed = self.transform(image=image)
        #    image = transformed['image'] 
     
        # if not torch.is_tensor(image):
        #     image = torch.from_numpy(image)
        
            
        return image, label


def create_train_val_loaders(train_images, train_labels, val_size=0.15, batch_size=batch_size, num_workers=num_workers, seed=42):
    torch.manual_seed(seed)
    dataset_size = len(train_labels)
    val_length = int(dataset_size * val_size)
    train_length = dataset_size - val_length
    
    full_train_dataset = CarDataset(
        train_images, 
        train_labels,
    )
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # train_subset.dataset.transform = train_transform 
    

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        # sampler=create_sampler_with_weighted_sampler(train_subset.dataset.labels),
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    print("\n" + "="*50)
    print("Data Split Information")
    print("="*50)
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_length}")
    print(f"Validation set size: {val_length}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print("="*50 + "\n")
    
    return train_loader, val_loader
