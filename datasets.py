import os
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def download_and_extract(data_dir):
    url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
    tgz_path = os.path.join(data_dir, '17flowers.tgz')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'jpg')):
        print('Downloading Flowers17 dataset...')
        urllib.request.urlretrieve(url, tgz_path)
        print('Extracting...')
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=data_dir)
        os.remove(tgz_path)
        print('Done!')
    else:
        print('Flowers17 dataset already exists.')

def get_data_loaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Flowers17原始数据没有子文件夹，需重组为ImageFolder格式
    img_dir = os.path.join(data_dir, 'jpg')
    foldered_dir = os.path.join(data_dir, 'flowers17_imgfolder')
    if not os.path.exists(foldered_dir):
        print('Reorganizing images into class folders...')
        os.makedirs(foldered_dir)
        for i in range(17):
            os.makedirs(os.path.join(foldered_dir, str(i)), exist_ok=True)
        for idx in range(1, 1361):
            img_name = f'image_{idx:04d}.jpg'
            class_idx = (idx - 1) // 80
            src = os.path.join(img_dir, img_name)
            dst = os.path.join(foldered_dir, str(class_idx), img_name)
            if not os.path.exists(dst):
                os.link(src, dst)
    dataset = datasets.ImageFolder(foldered_dir, transform=train_transform)
    total = len(dataset)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    val_set.dataset.transform = test_transform
    test_set.dataset.transform = test_transform
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = './data'
    download_and_extract(data_dir)
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)
    print('Data loaders are ready!') 