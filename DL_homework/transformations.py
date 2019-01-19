import torchvision.transforms as transforms

def basic():
    transform = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transform


def augmentation1():
    transform = transforms.Compose(
        [
            transforms.Resize((40,40)),
            transforms.RandomCrop(size = (32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


def augmentation2():
    transform = transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    return transform


def augmentation3():
    transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    return transform