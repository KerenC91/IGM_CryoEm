import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, all_reduce
import os
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time 

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" #any free port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    
def reduce_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        dist.get_world_size() if dist.is_initialized() else 1
    )
    return rt


class NetBN(nn.Module):
    def __init__(self, gpu_id):
        super(NetBN, self).__init__()
        self.gpu_id = gpu_id
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.SyncBatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.SyncBatchNorm(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.SyncBatchNorm(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
class Net(nn.Module):
    def __init__(self, gpu_id):
        super(Net, self).__init__()
        self.gpu_id = gpu_id
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_imgs: int,
        suffix: str,
        gpu_id: int,
        save_every: int,
        world_size: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.num_imgs = num_imgs
        self.suffix = suffix
        self.model = DDP(model, device_ids=[gpu_id])
        self.world_size = world_size

    def _run_batch(self, source, targets):
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        return loss

    def _run_epoch(self, epoch):
        loss_sum = 0
        b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.optimizer.zero_grad()

        i = 0
        for source, targets in self.train_data: 
            # Move to device
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            loss = self._run_batch(source, targets)   
            loss_sum += loss
            #print(f"[GPU{self.gpu_id}] Epoch {epoch} iter {i}| loss: {loss} loss_sum: {loss_sum}")
            i += 1
        loss_sum *= b_sz
        loss_sum.backward()
        self.optimizer.step()
        return loss_sum

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"./figures/checkpoint_{self.suffix}.pt"
        torch.save(ckp, PATH)
        #print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        loss_list = []
        b_sz = len(next(iter(self.train_data))[0])
        for epoch in range(max_epochs):
            loss_sum = self._run_epoch(epoch)
            # Get loss from all processes
            all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            #print(f"[GPU{self.gpu_id}] Epoch {epoch} | loss_sum: {loss_sum} after all_reduce")

            # Only gpu 0 operating now...
            if self.gpu_id == 0: 
                loss_list.append(loss_sum.item() / (self.world_size * len(self.train_data) * b_sz))
            # Save checkpoint
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        # save data    
        if self.gpu_id == 0:
            b_sz = len(next(iter(self.train_data))[0])
            #print(f'#loss points {len(loss_list)}')
            np.save(f'./figures/loss_ddp_{self.num_imgs}im_{b_sz}bs_{max_epochs}epochs_{self.world_size}gpus_{self.suffix}.npy', loss_list)
            plt.figure()
            plt.plot(loss_list)
            plt.savefig(f'./figures/loss_ddp_{self.num_imgs}im_{b_sz}bs_{max_epochs}epochs_{self.world_size}gpus_{self.suffix}.png')
            plt.close()



def load_train_objs(num_imgs, rank, world_size):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('../../data', train=True, download=False,
                       transform=transform)
    class_idx = 8
    idx = train_set.targets==class_idx
    train_set.targets = train_set.targets[idx]
    train_set.data = train_set.data[idx] 
    
    train_set.data = train_set.data[0:num_imgs]
    
    model = Net(rank)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3 * world_size)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, num_imgs: int, suffix: str):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(num_imgs, rank, world_size)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, num_imgs, suffix, rank, save_every, world_size)
    # Get start time
    if trainer.gpu_id not in [-1, 0]:
        dist.barrier()
    # Only gpu 0 operating now...
    if trainer.gpu_id == 0:
        print(f"Running {os.path.basename(__file__)}"
              f" with {world_size} gpus, "
              f"{total_epochs} total epochs, "
              f"{num_imgs} images, "
              f"{batch_size} batch size, "
              f"save checkpoint every {save_every} epochs")
        
        start_time = time.time() 
        dist.barrier()
    
    trainer.train(total_epochs)
    
    # Get end time 
    if trainer.gpu_id not in [-1, 0]:
        dist.barrier()

    if trainer.gpu_id == 0:
        # Only gpu 0 operating now...
        end_time = time.time()  
        print(f"Time taken to train in {os.path.basename(__file__)}: {end_time - start_time} seconds")
        dist.barrier()

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=100, type=int, help='Total epochs to train the model (default: 100)')
    parser.add_argument('--save_every', default=30, type=int, help='How often to save a snapshot. (default: 30)')
    parser.add_argument('--batch_size', default=None, type=int, help='Input batch size on each device (default: 1)')
    parser.add_argument('--num_imgs', default=5, type=int, help='Input batch size on each device (default: 5)')
    parser.add_argument('--suffix', default='', type=str, help='suffix, default is ''')
    parser.add_argument('--nproc', default=torch.cuda.device_count(), type=int, help='nproc, default is the number of available gpus on the machine')

    args = parser.parse_args()
    world_size = torch.cuda.device_count() # How many GPUs are availale on a machine
    # rank is the device
    if args.nproc:
        world_size = args.nproc
        
    if args.batch_size is None:
        args.batch_size = int(args.num_imgs / world_size)
    # Otherwise value is set as the user provided
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.num_imgs, args.suffix), nprocs=world_size)
