import glob
from os.path import join
from torch.utils.data import Dataset
from matplotlib.image import imread


class MRI(Dataset):
    """MRI brain tumor scan dataset.

    Download the dataset from here: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?resource=download
    Extract the zip file in a directory called "mri"
    """

    def __init__(self, root, transform=None):
        super(MRI, self).__init__()
        self.transform = transform
        self.paths = glob.glob(join(root, "mri", "**", "*.jpg"), recursive=True)

    def __getitem__(self, index):
        img = imread(self.paths[index])
        if self.transform is not None:
            return self.transform(img), None
        return img, None

    def __len__(self):
        return len(self.paths)
