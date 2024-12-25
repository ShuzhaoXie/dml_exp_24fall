import os
from urllib import parse
import torch
from torch.distributed.distributed_c10d import get_rank, get_world_size
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import argparse
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync
import dist_utils


n0 = 0
n1 = 0
n2 = 0
d0 = f'cuda:{n0}'
d1 = f'cuda:{n1}'
d2 = f'cuda:{n2}'

class SubNetConv(nn.Module):
    def __init__(self, in_channels, device):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2
        ).to(device)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        ).to(device)
        self.device = device

    def forward(self, x_rref):
        x = x_rref.to(self.device)  # 将远程引用的数据移动到本地
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        return x

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class SubNetFC(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to(device)
        self.fc2 = nn.Linear(120, num_classes).to(device)
        self.device = device

    def forward(self, x_rref):
        x = x_rref.to(self.device)  # 将远程引用的数据移动到本地
        # x = x.view(x.size(0), -1)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class ParallelNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, device=d0):
        super().__init__()
        # 分别远程声明SubNetConv和SubNetFC
        self.subnet_conv = rpc.remote("worker1", SubNetConv, args=(in_channels, d1))
        self.subnet_fc = rpc.remote("worker2", SubNetFC, args=(num_classes, d2))
        self.device = device

    def forward(self, x):
        x = self.subnet_conv.rpc_sync().forward(x)
        x = self.subnet_fc.rpc_sync().forward(x)
        return x

    def parameter_rrefs(self):
        conv_params = self.subnet_conv.rpc_sync().parameter_rrefs()
        fc_params = self.subnet_fc.rpc_sync().parameter_rrefs()
        return conv_params + fc_params


def train(model, dataloader, loss_fn, optimizer, num_epochs=2):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.0
    model.train()
    dist_utils.init_parameters(model)
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            with dist_autograd.context() as context_id:
                inputs, labels = batch_data
                # optimizer.zero_grad()
                inputs = inputs.to(d0)
                labels = labels.to(d0)
                output = model(inputs)
                loss = loss_fn(output, labels)
                dist_autograd.backward(context_id, [loss])
                optimizer.step(context_id)

            loss_total += loss.item()

            if i % 20 == 19:
                print(
                    "Device: %d epoch: %d, iters: %5d, loss: %.3f"
                    % (dist_utils.get_local_rank(), epoch + 1, i + 1, loss_total / 20)
                )
                loss_total = 0.0

    print("Training Finished!")


def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs.to(d0)
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.cpu().eq(labels.data.view_as(pred)).sum().item()
    print(
        "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
            correct, size, 100 * correct / size
        )
    )


def main():
    args = parse_args()
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    DATA_PATH = "./data"
    
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    
    if args.rank == 0:    
        options.set_device_map("worker1", {n0: n1})
        options.set_device_map("worker2", {n0: n2})
        rpc.init_rpc("worker0", rank=args.rank, world_size=args.n_devices, rpc_backend_options=options)
        # construct the model
        model = ParallelNet(in_channels=1, num_classes=10, device=d0)
        # construct the dataset
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(
            DATA_PATH, train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            DATA_PATH, train=False, download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=32, shuffle=False
        )

        # construct the loss_fn and optimizer
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        dist_optimizer = DistributedOptimizer(
            torch.optim.SGD, model.parameter_rrefs(), lr=0.01
        )

        train(model, train_loader, loss_fn, dist_optimizer)
        test(model, test_loader)

    elif args.rank == 1:
        rpc.init_rpc("worker1", rank=args.rank, world_size=args.n_devices, rpc_backend_options=options)
        print("Training on the worker1...")

    elif args.rank == 2:
        rpc.init_rpc("worker2", rank=args.rank, world_size=args.n_devices, rpc_backend_options=options)
        print("Training on the worker2...")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_devices", default=1, type=int, help="The distributd world size."
    )
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument(
        "--master_addr", default="localhost", type=str, help="ip of rank 0"
    )
    parser.add_argument("--master_port", default="60001", type=str, help="ip of rank 0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()