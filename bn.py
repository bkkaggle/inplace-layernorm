import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from models import Net

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree


def extract(elem, tag, drop_s):
    text = elem.find(tag).text
    if drop_s not in text:
        raise Exception(text)
    text = text.replace(drop_s, "")
    try:
        return int(text)
    except ValueError:
        return float(text)


def memusage():
    i = 0

    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816:
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    print('\n\nUpdated at %s\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n%s\n\n' % (
        now, d["gpu_util"], d["mem_used_per"], msg))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--abn_type', type=str)
    parser.add_argument('--jit_script', default=False, action="store_true")

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

    model = Net(args.abn_type).to(device)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(0, 2):
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

            if i == 10:
                memusage()

            if args.fast and i > 10:
                break

        correct = 0

        model.eval()
        with torch.no_grad():
            for j, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                memusage()
                data, target = data.to(device), target.to(device)

                output = model(data)
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
