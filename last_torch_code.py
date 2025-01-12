import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from modules import dataHandler, dataprocessing, models
from modules import scalers as scaling
import matplotlib.pyplot as plt
import pickle as pkl
from multiprocessing import Process, Manager, Lock
import numpy as np
from torch.utils.data import TensorDataset
from torch.cuda import device_count, get_device_name, get_device_properties, max_memory_allocated, max_memory_cached

def ddp_setup():
    init_process_group(backend="nccl")
    gpu_id=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu_id)
    return gpu_id
        
def print_gpu_info():
    num_gpus = device_count()
    print(f"Total {num_gpus} CUDA Capable device(s)\n")
    
    for gpu_id in range(num_gpus):
        device = get_device_name(gpu_id)
        properties = get_device_properties(gpu_id)

        print(f"Device {gpu_id}: \"{device}\"")
        print(f"Total amount of global memory: {properties.total_memory / 1024**3:.2f} GB")
        print(f"(080) Multiprocessors, (064) CUDA Cores/MP: {properties.multi_processor_count}, {properties.multi_processor_count * 64} CUDA Cores")

    
class YourModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.fc2 = nn.Linear(64, 128)         
        self.fc3 = nn.Linear(128, 256)        
        self.fc4 = nn.Linear(256, 512)        
        self.fc5 = nn.Linear(512, output_size) 

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))  
        x = F.relu(self.fc4(x))  
        x = self.fc5(x)         
        return x
    
class Trainer:
    def __init__(
        self,
        model: YourModel,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        early_stopping_patience: int,
        scaler_X: MinMaxScaler,
        scaler_y1: MinMaxScaler,
        scaler_y2: MinMaxScaler,
        min_len: int
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.scaler_X = scaler_X
        self.scaler_y1 = scaler_y1
        self.scaler_y2 = scaler_y2
        self.min_len = min_len
        
        self.avg_temp_losses=[]
        self.avg_salinity_losses=[]

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.best_val_loss = snapshot.get("BEST_VAL_LOSS", float('inf'))
        self.patience_counter = snapshot.get("PATIENCE_COUNTER", 0)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()

        output = self.model(source.clone().detach().requires_grad_(True)).to(self.gpu_id)
        output = output.view(-1, 2, self.min_len)
        targets = targets.view(-1, 2, self.min_len)
        loss = torch.sqrt(F.mse_loss(output, targets))

        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            
        val_loss_temperature, val_loss_salinity = self._validate(epoch)
        
        self.avg_temp_losses.append(val_loss_temperature)
        self.avg_salinity_losses.append(val_loss_salinity)
        
        if epoch % 100 == 0:
            avg_temp_loss = sum(self.avg_temp_losses) / len(self.avg_temp_losses)
            avg_salinity_loss = sum(self.avg_salinity_losses) / len(self.avg_salinity_losses)
            print(f"[GPU{self.gpu_id}] | Epoch {epoch} | Average Temperature Validation Loss : {avg_temp_loss:.6f} | Average Salinity Validation Loss : {avg_salinity_loss:.6f} | Loss Function: RMSE | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    
    def _validate(self, epoch):
        self.model.eval()
        val_loss_temperature = 0.0
        val_loss_salinity = 0.0
        total = 0

        with torch.no_grad():
            for source, targets in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source.clone().detach().requires_grad_(True)).to(self.gpu_id)
                output = output.view(-1, 2, self.min_len)
                targets = targets.view(-1, 2, self.min_len)

                # RMSE hesaplamalarını yapın
                for i in range(2):
                    targets_column = targets[:, i, :]
                    output_column = output[:, i, :]
                    val_loss = torch.sqrt(torch.mean((targets_column - output_column) ** 2))
                    if i == 0:
                        val_loss_temperature += val_loss.item()
                    else:
                        val_loss_salinity += val_loss.item()
                    

        val_loss_temperature /= len(self.val_data)
        val_loss_salinity /= len(self.val_data)

        self.model.train()
        return val_loss_temperature, val_loss_salinity


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "BEST_VAL_LOSS": self.best_val_loss, 
            "PATIENCE_COUNTER": self.patience_counter
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)

            if epoch % self.save_every == 0 and epoch !=0:
                avg_temp_loss, avg_salinity_loss = self._validate(epoch)
                val_loss = avg_temp_loss + avg_salinity_loss

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_snapshot(epoch)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
                    
                if epoch %100==0:
                    self.avg_temp_losses=[]
                    self.avg_salinity_losses=[]
                    
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
def load_train_objs(gpu_id, learning_rate):
    model = YourModel(input_size=18, output_size=30).to(gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return  model, optimizer


def load_and_preprocess_data(min_len, gpu_id, batch_size):
    current_directory = os.getcwd()
    print(current_directory)
    # ts_profiles = dataHandler.load_all(verbose=True)[1]
    file_path = os.path.join(current_directory, 'backup/profiles.pkl')
    with open(file_path, 'rb') as file:
        ts_profiles = pkl.load(file)

    X = []  # Create an empty list for input data
    y = []  # Create an empty list for target data

    for profile in ts_profiles:
        depth = profile.depth
        depth_length = len(depth)

        if depth_length <= min_len:
            continue
        else:
            depth = depth[:min_len]
            depth_float = list(map(float, depth))
            data = [profile.time, profile.point.x, profile.point.y] + depth_float
            X.append(torch.tensor(data, dtype=torch.float32))
            y.append([profile.temperatur[:min_len], profile.salinity[:min_len]])

    X = torch.stack(X)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y1 = MinMaxScaler()
    scaler_y2 = MinMaxScaler()
    scaler_y1.fit(y_train.view(y_train.size(0), -1)[:, :min_len])
    scaler_y2.fit(y_train.view(y_train.size(0), -1)[:, min_len:])
    y_train_scaled1 = scaler_y1.transform(y_train.view(y_train.size(0), -1)[:, :min_len])
    y_train_scaled2 = scaler_y2.transform(y_train.view(y_train.size(0), -1)[:, min_len:])
    y_val_scaled1 = scaler_y1.transform(y_val.view(y_val.size(0), -1)[:, :min_len])
    y_val_scaled2 = scaler_y2.transform(y_val.view(y_val.size(0), -1)[:, min_len:])
    y_test_scaled1 = scaler_y1.transform(y_test.view(y_test.size(0), -1)[:, :min_len])
    y_test_scaled2 = scaler_y2.transform(y_test.view(y_test.size(0), -1)[:, min_len:])

    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device=gpu_id)
    X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32, device=gpu_id)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device=gpu_id)

    y_train_scaled1 = torch.tensor(y_train_scaled1, dtype=torch.float32, device=gpu_id)
    y_train_scaled2 = torch.tensor(y_train_scaled2, dtype=torch.float32, device=gpu_id)
    y_val_scaled1 = torch.tensor(y_val_scaled1, dtype=torch.float32, device=gpu_id)
    y_val_scaled2 = torch.tensor(y_val_scaled2, dtype=torch.float32, device=gpu_id)
    y_test_scaled1 = torch.tensor(y_test_scaled1, dtype=torch.float32, device=gpu_id)
    y_test_scaled2 = torch.tensor(y_test_scaled2, dtype=torch.float32, device=gpu_id)

    y_train_scaled = torch.cat((y_train_scaled1, y_train_scaled2), dim=1).view(y_train.size(0), 2, min_len).to(gpu_id)
    y_val_scaled = torch.cat((y_val_scaled1, y_val_scaled2), dim=1).view(y_val.size(0), 2, min_len).to(gpu_id)
    y_test_scaled = torch.cat((y_test_scaled1, y_test_scaled2), dim=1).view(y_test.size(0), 2, min_len).to(gpu_id)

    train_set = TensorDataset(X_train_scaled, y_train_scaled)
    val_set = TensorDataset(X_val_scaled, y_val_scaled)
    test_set = TensorDataset(X_test_scaled, y_test_scaled)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, train_set, val_set, test_set, scaler_X, scaler_y1, scaler_y2

  
def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", early_stopping_patience: int = 5):
    ddp_setup()
    
    gpu_id=int(os.environ["LOCAL_RANK"])
    print_gpu_info()
    
    batch_size=64
    min_len=15
    total_epochs=500
    input_size=18
    output_size=30
    learning_rate=0.001
    save_every=50
    
    my_model= YourModel(input_size,output_size)
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, train_set, val_set, test_set, scaler_X, scaler_y1, scaler_y2 = load_and_preprocess_data(min_len, gpu_id, batch_size)
    model, optimizer = load_train_objs(gpu_id, learning_rate)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)
    trainer = Trainer(my_model, train_data, val_data, optimizer, save_every, snapshot_path, early_stopping_patience, scaler_X, scaler_y1, scaler_y2, min_len)
    trainer.train(total_epochs)
    avg_temp_loss, avg_salinity_loss = trainer._validate(total_epochs)

    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--snapshot_path', default="snapshot.pt", type=str, help='Path to save snapshots (default: snapshot.pt)')
    parser.add_argument('--early_stopping_patience', default=5, type=int, help='Early stopping patience (default: 5)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.snapshot_path, args.early_stopping_patience)

