import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml

def load_checkpoints(path1, path2):
    model1 = Net()
    model2 = Net()

    checkpoint1 = torch.load(path1)
    checkpoint2 = torch.load(path2)
    # with open('checkpoint1.yaml', 'w') as file:
    #     # Write the data to the YAML file
    #     yaml.dump(checkpoint1, file)
        
    # with open('checkpoint2.yaml', 'w') as file:
    #     # Write the data to the YAML file
    #     yaml.dump(checkpoint2, file)        
    model1.load_state_dict(checkpoint1)
    model2.load_state_dict(checkpoint2)
    i = 0
    print(f'{path1}')
    for param_tensor1, param_tensor2 in zip(model1.state_dict(), model2.state_dict()):
        print(param_tensor1, "\t", model1.state_dict()[param_tensor1].size())
        print(param_tensor2, "\t", model2.state_dict()[param_tensor2].size())
        if model1.state_dict()[param_tensor1].size() != model2.state_dict()[param_tensor2].size():
            print("error")
        diff = model1.state_dict()[param_tensor1] - model2.state_dict()[param_tensor2]
        for var in diff:
            if abs(var) > 0:
                print(f'{param_tensor2} var')
        print(f'{param_tensor1} and {param_tensor2} are not identical. diff={diff}')


class NetBN(nn.Module):
    def __init__(self, gpu_id):
        super(NetBN, self).__init__()
        self.gpu_id = gpu_id
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)
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
    def __init__(self):
        super(Net, self).__init__()
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
        suffix: str
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.num_imgs = num_imgs
        self.suffix = suffix
        
    def _run_batch(self, source, targets):
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        return loss

    def _run_epoch(self, epoch):
        loss_sum = 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.optimizer.zero_grad()
        for source, targets in self.train_data:
            loss_sum += self._run_batch(source, targets)
        print(f"GPU0 Epoch {epoch} | loss_sum: {loss_sum}")
        loss_sum *= b_sz
        loss_sum.backward()
        self.optimizer.step()
        loss_sum /= len(self.train_data) * b_sz
        return loss_sum

    def train(self, max_epochs: int):
        loss_list = []
        for epoch in range(0, max_epochs):
            loss_sum = self._run_epoch(epoch)
            loss_list.append(loss_sum.item())            
        # Save data    
        b_sz = len(next(iter(self.train_data))[0])
        print(f'#loss points {len(loss_list)}')
        np.save(f'./figures/loss_{self.num_imgs}im_{b_sz}bs_{max_epochs}epochs_1gpu_{self.suffix}.npy', loss_list)
        plt.figure()
        plt.plot(loss_list)
        plt.savefig(f'./figures/loss_{self.num_imgs}im_{b_sz}bs_{max_epochs}epochs_1gpu_{self.suffix}.png')
        plt.close()



def load_train_objs(num_imgs):
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

    train_set.data = train_set.data[:num_imgs]
    model = Net()  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False
    )


def main(total_epochs: int, batch_size: int, num_imgs: int, suffix: str):
    dataset, model, optimizer = load_train_objs(num_imgs)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, num_imgs, suffix)
    start_time = time.time()
    trainer.train(total_epochs)
    end_time = time.time()
    print(f"Time taken to train in {os.path.basename(__file__)}:", 
          end_time - start_time, "seconds")

def plot_all(folder):
    plt.figure()
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            data = np.load(os.path.join(folder, filename))
            info = os.path.splitext(filename)[0]
            info = info.replace("_", " ")
            plt.plot(data, label=info)
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('normalized loss')
    plt.title('training loss')
    plt.savefig(f'{folder}/loss_all_.png')
    plt.close()    
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple training job')
    parser.add_argument('--total_epochs', default=100, type=int, help='Total epochs to train the model (default: 1000)')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 1)')
    parser.add_argument('--num_imgs', default=4, type=int, help='Input batch size on each device (default: 5)')
    parser.add_argument('--suffix', default='', type=str, help='Suffix, default is ''')
    parser.add_argument('--print', default=False, type=bool, help='Weather to print debug or not. Default is False')

    args = parser.parse_args()
    # load_checkpoints('./figures/checkpoint_ddp_24im_parallel_bs1_2gpu_norm_1epoch.pt',
    #                  './figures/checkpoint_24im_bs1_1epoch_unnorm.pt')
    plot_all(folder='./figures/loss_norm_after_back')
    main(args.total_epochs, args.batch_size, args.num_imgs, args.suffix)
