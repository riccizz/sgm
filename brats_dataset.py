import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


class Brats(data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.img_list = os.listdir("../ddpm/dataset/brats")
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = np.load(f'../ddpm/dataset/brats/brats_{index}.npy')
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    import tqdm
    dataset = Brats(transform=transforms.ToTensor())
    print(len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897])
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
    for x in train_loader:
        print(x.shape)
        break

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
        )
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(58982400, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1024, 10)
        )
        def forward(self, x):
            x = self.conv1(x)

            x = x.view(-1, 58982400)
            x = self.dense(x)
            return x

    device_ids = [0, 1]
    model = Model()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 50
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
        for data in tqdm.tqdm(train_loader):
            X_train = data
            X_train = X_train.cuda()
            outputs = model(X_train)
            _,pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, outputs)
    
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            running_correct += torch.sum(pred == y_train.data)


