import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import deeptime as dt
import mdshare

# data loading
X = np.load(mdshare.fetch('alanine-dipeptide-3x250ns-backbone-dihedrals.npz'))

dihedrals = [
    np.load(mdshare.fetch('alanine-dipeptide-3x250ns-backbone-dihedrals.npz'))[f'arr_{i}'] for i in range(3)
]
coordinates = [
    np.load(mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-positions.npz'))[f'arr_{i}'] for i in range(3)
]

# Tau, how much is the timeshift of the two datasets
tau = 1

# Which trajectory points percentage is used as training
train_ratio = 0.9

concat_coords = coordinates[0] # np.concatenate(coordinates)
concat_dih = dihedrals[0] # np.concatenate(dihedrals)
print(concat_coords.shape, concat_dih.shape)

# separate the data into training and validation
n_train = int(np.floor(len(concat_coords) * train_ratio))
n_validation = len(concat_coords) - tau - n_train

# separate the time-lagged datapoints and shuffle data
dataset = dt.data.TimeLaggedDataset.from_trajectory(tau, concat_coords.astype(np.float32))
train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_validation])

# check the availability of cuda
assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(12)

#from torch_geometric.nn import SchNet

class VAMPSchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Here we initilze six layers of fully connected network.
        """
        super(VAMPSchNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))
        x = F.elu(self.linear4(x))
        x = F.elu(self.linear5(x))
        x = F.softmax(self.linear6(x), dim=1)

        return x




# define the training function
def train_with_sampler(n_epochs, batch_size=1024, learning_rate=1e-4):
    from torch.utils.data import RandomSampler, BatchSampler

    # define layer sizes
    input_size = concat_coords.shape[1]
    hidden_size = 128
    output_size = 6
 
    lobe = VAMPSchNet(input_size, hidden_size, output_size)
    lobe = lobe.to(device)    
    
    vnet = dt.decomposition.VAMPNet(lobe, device=device, optimizer='Adam', learning_rate=learning_rate)
    
    sampler = BatchSampler(RandomSampler(train_data, True), batch_size=batch_size, drop_last=True)
    loader = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*train_data[:]), 
                                         drop_last=True, batch_size=1, sampler=sampler)

    sampler_val = BatchSampler(RandomSampler(val_data, True), batch_size=batch_size, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*val_data[:]), 
                                         drop_last=True, batch_size=1, sampler=sampler_val)
    
    for epoch in range(n_epochs):

        lobe.train()
        for batch_0, batch_t in loader:
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            vnet.partial_fit((batch_0, batch_t))

        lobe.eval()
        scores_val = []
        for batch_0, batch_t in loader_val:
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            
            score_val = vnet.validate((batch_0, batch_t))
            scores_val.append(score_val.cpu().numpy())
        print(f"    Epoch {epoch+1}/{n_epochs}: Validation score {np.mean(scores_val):.4f}", end='\r')
    return vnet, scores_val
vnet, scores_val = train_with_sampler(50, learning_rate=5e-4)  # or the easier version, train(...)

train_scores = vnet.train_scores.T # numpy array with [[index] [score]]
val_scores = np.array([np.array(np.arange(len(scores_val))), scores_val])

# output the training and validation scores into npy files
np.save('vn_train_scores.npy', train_scores)
np.save('vn_val_scores.npy', val_scores)

