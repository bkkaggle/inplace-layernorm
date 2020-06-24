import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


class BnModel(nn.Module):
    def __init__(self):
        super(BnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fast', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")

    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    model = BnModel().to(device)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        train_loss = 0
        val_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, (data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if args.fast and i > 10:
                break

        correct = 0

        model.eval()
        with torch.no_grad():
            for j, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = F.nll_loss(output, target, reduction='sum').item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss += loss

        train_loss /= (i + 1)
        val_loss /= (j + 1)

        print(
            f"loss: {train_loss}, val_loss: {val_loss}, val_acc: {correct / (j + 1)}")


if __name__ == "__main__":
    main()
