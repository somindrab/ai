import pathlib

imgdir_path=pathlib.Path('images/cats_and_dogs')

file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

print(file_list)


class ImageDataset(Dataset):
    def __init(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)

