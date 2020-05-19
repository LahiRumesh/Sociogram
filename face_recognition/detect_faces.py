from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import os

data_dir = 'faces'

batch_size = 32
workers = 0 if os.name == 'nt' else 8


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True
    )

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_detect'))
        for p, _ in dataset.samples
]
      
loader = DataLoader(
    dataset,
    num_workers=0,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)


for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)

