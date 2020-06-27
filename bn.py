import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock

import wandb

from models import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resnet', default=False, action="store_true")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--abn_type', type=str)

    parser.add_argument('--fast', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--tags', nargs='+')

    args = parser.parse_args()

    wandb.init(project="inplace-layernorm", config=args, tags=args.tags)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    if args.resnet:

        if args.abn_type == 'pt':
            abn = nn.BatchNorm2d
        elif args.abn_type == 'custom':
            abn = BatchNorm

        elif args.abn_type == 'autograd':
            abn = BatchNormAutograd

        elif args.abn_type == 'checkpoint':
            abn = CheckpointBN

        model = ResNet(BasicBlock, [2, 2, 2, 2],
                       num_classes=10, norm_layer=abn).to(device)

    else:
        model = Net(args.abn_type).to(device)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           transforms.Lambda(
                               lambda x: torch.cat([x, x, x], dim=0))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(
                lambda x: torch.cat([x, x, x], dim=0))
        ])),
        batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(0, args.epochs):
        train_loss = 0
        val_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, (data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            output = F.log_softmax(output, dim=1)

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
                output = F.log_softmax(output, dim=1)

                loss = F.nll_loss(output, target, reduction='sum').item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss += loss

        train_loss /= (i + 1)
        val_loss /= (j + 1)

        print(
            f"loss: {train_loss}, val_loss: {val_loss}, val_acc: {correct / len(test_dataloader.dataset)}")


if __name__ == "__main__":
    main()
