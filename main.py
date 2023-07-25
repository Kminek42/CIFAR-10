import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import learning_time_estimation as lte
import time
import model_class as mc

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def prepare_dataset(*, training):    
    if training:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(size=32, padding=6),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="./downloads",
            train=True,
            download=True,
            transform=transform
        )

        train_dataset, validate_set = torch.utils.data.random_split(train_dataset, [40000, 10000])
        print("Train dataset length:", len(train_dataset))
        print("Validate dataset length:",len(validate_set))

        return train_dataset, validate_set

    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])

        test_dataset = torchvision.datasets.CIFAR10(
            root="./downloads",
            train=False,
            download=True,
            transform=transform
        )

        print("Test dataset length:", len(test_dataset))

        return test_dataset

def validate(*, model, loader, dev):
    all = 0
    good = 0
    for inputs, targets in iter(loader):
        inputs, targets = inputs.to(dev), targets.to(dev)
        outputs = model.forward(inputs)
        for i in range(len(outputs)):
            if torch.argmax(outputs[i]) == targets[i]:
                good += 1
            
            else:
                img = torchvision.transforms.ToPILImage()(inputs[i])
                # plt.imshow(img)
                # plt.title(f"Real: {labels[int(targets[i])]};  NN: {labels[int(torch.argmax(outputs[i]))]}")
                # plt.show()
            all += 1

    return (good / all)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
train = False
if train:
    train_dataset, validate_dataset = prepare_dataset(training=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    validate_loader = DataLoader(
    dataset=validate_dataset,
    batch_size=64,
    shuffle=True
    )

    model = mc.ResNet().to(dev)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_n = 20
    t0 = time.time()

    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for inputs, targets in iter(train_loader):
            inputs, targets = inputs.to(dev), targets.to(dev)
            outputs = model.forward(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss

        print(f"\nEpoch: {epoch}, mean loss: {loss_sum / len(train_loader)}")

        acc = validate(model=model, loader=validate_loader, dev=dev)
        print(f"Validation accuracy: {acc}")

        lte.show_time(t0, epoch / epoch_n)

        stats_file = open(file="stats.csv", mode="a")
        stats_file.write(f"{loss_sum / len(train_loader)},{acc}\n")
        stats_file.close()
    
    torch.save(obj=model.to("cpu"), f="model.pt")

else:
    test_dataset = prepare_dataset(training=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )
    print(len(test_dataset))
    model = torch.load(f="model.pt").to(dev)

    result = validate(model=model, loader=test_loader, dev=dev)
    print(result)
