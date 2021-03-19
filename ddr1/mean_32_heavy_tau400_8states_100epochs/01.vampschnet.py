import numpy as np
from math import pi as PI
import ase
import torch
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import torch.nn.functional as F
from torch_geometric.nn import SchNet, radius_graph, MessagePassing, global_mean_pool
import deeptime as dt
import mdshare
import mdtraj as md
from numpy.random import default_rng
#import sys
#np.set_printoptions(threshold=sys.maxsize)

# Tau, how much is the timeshift of the two datasets
tau = 400 # 0.25 ns * 400 = 100 ns (0.25 ns timesteps)
# Which trajectory points percentage is used as training
train_ratio = 0.85

# trajectorie with DFG flips
traj_indices = [(42, 1), (42, 3), (42, 6), (48, 11), (48, 12), (48, 13), (53, 10), (53, 18), (53, 4), (54, 14), (55, 13), (57, 11), (62, 18), (62, 9), (67, 19), (67, 6), (69, 11), (69, 3), (69, 7), (70, 17), (71, 19), (71, 7), (73, 10), (73, 15), (74, 10), (74, 14), (74, 16), (74, 19), (74, 6), (75, 19), (77, 17)] 
#traj_indices = [(42, 1), (42, 3), (42, 6), (48, 11), (48, 12)]
#traj_indices = [(42, 1), (42, 3), (42, 6), (48, 11), (48, 12), (48, 13), (53, 10), (53, 18), (53, 4), (54, 14), (55, 13), (57, 11), (62, 18), (62, 9), (67, 19), (67, 6), (69, 11), (69, 3), (69, 7), (70, 17), (71, 19), (71, 7), (73, 10), (73, 15), (74, 10)]

# select trajectories for training and testing
num_trajs = len(traj_indices) # total number of trajs
test_index = list()
for run in range(40, 80):
    for clone in range(20):
        if (run, clone) not in traj_indices:
            test_index.append((run, clone))
np.save(f'train_index.npy', np.array(traj_indices))
np.save(f'test_index.npy', np.array(test_index))


# find the relevant atom indices
key_res = list()
key_res.append(np.arange(184, 205)) # A-loop: 21 aa (M184 - I204)
key_res.append(np.arange(14, 20)) # P-loop: 6 aa (G14 - G19)
key_res.append(np.arange(181, 183)) # DFG-flip: D181, F182
key_res.append(np.arange(65, 76)) # aC helix: 11 aa (D65 - R75)

key_res = list(i for sublist in key_res for i in sublist)

traj = md.load(f'../data/run{traj_indices[0][0]}-clone{traj_indices[0][1]}.h5')
topology = traj.topology

table, bonds = topology.to_dataframe() # JG debug
atoms = table.values
#print(atoms)

atom_indices = list()
for res in key_res:
    #atom_indices.append(topology.select(f"chainid 0 and residue {res} and name == CA"))
    atom_indices.append(topology.select(f"chainid 0 and residue {res} and type != H"))

atom_indices = list(i for sublist in atom_indices for i in sublist)
print("len of features: ", len(atom_indices))
num_nodes = len(atom_indices)

# get the atom typs 
atom_type = list()
for i in atom_indices:
    if atoms[i][2] == 'C':
        atom_type.append(0)
    elif atoms[i][2] == 'O':
        atom_type.append(1)
    elif atoms[i][2] == 'N':
        atom_type.append(2)
    else:
        atom_type.append(3)
print("atom type: ", atom_type)

# featurize the selected trajectories
list_train_data = list()
list_train_lag = list()
list_val_data = list()
list_val_lag = list()
list_train_indices = list()
list_val_indices = list()

for index in traj_indices:
    print(f"traj index: {index}")
    traj = md.load(f'../data/run{index[0]}-clone{index[1]}.h5')
    topology = traj.topology
    pos = traj.xyz[:,atom_indices,:] # pos is a np.array of shape (data_len, num_nodes, 3) 
    # separate the data into training and validation
    n_train = int(np.floor(len(pos) * train_ratio))
    n_validation = len(pos) - tau - n_train
    print("JG debug: ")
    print(n_train, n_validation)
    # separate the time-lagged datapoints and shuffle data
    dataset = dt.data.TimeLaggedDataset.from_trajectory(tau, pos.astype(np.float32))
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_validation])
    list_train_indices.append(train_data.indices)
    list_val_indices.append(val_data.indices)
    list_train_data.append(train_data.dataset.data)
    list_train_lag.append(train_data.dataset.data_lagged)
    list_val_data.append(val_data.dataset.data)
    list_val_lag.append(val_data.dataset.data_lagged)
total_train_ind = [i for sublist in list_train_indices for i in sublist]
total_val_ind = [i for sublist in list_val_indices for i in sublist]

total_train_data = torch.utils.data.Subset(dt.data.TimeLaggedDataset(np.array([i for sublist in list_train_data for i in sublist]), np.array([i for sublist in list_train_lag for i in sublist])), total_train_ind)
total_val_data = torch.utils.data.Subset(dt.data.TimeLaggedDataset(np.array([i for sublist in list_val_data for i in sublist]), np.array([i for sublist in list_val_lag for i in sublist])), total_val_ind)
print("len of train data: ", len(total_train_data))
#print("len of val data: ", len(total_val_data))

# check the availability of cuda
assert torch.cuda.is_available()
device = torch.device("cuda:0")
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(12)


# define some classes necessary for SchNet
class InteractionBlock(Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class CFConv(MessagePassing):
    #from torch.nn import Linear
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ShiftedSoftplus(Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

# define VAMPSchNet
class VAMPSchNet(Module):
    def __init__(self, hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, atomref, input_size, hidden_size, output_size):
        """
           
        The SchNet part of the code is based on the implementaion of SchNet (`"SchNet: A Continuous-filter 
        Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`) as 
        in torch_geometric.nn.models.schnet. The VAMPnet part of the code is based on the implementation of VAMPnet 
        (`"VAMPnets for deep learning of molecular kinetics" <https://www.nature.com/articles/s41467-017-02388-1>`) 
        as in https://deeptime-ml.github.io/notebooks/vampnets.html. 

        Example
        -------

        Notes
        -----

        Attributes
        ----------
        hidden_channels : int
                The hidden embedding size in SchNet (default: `128`)
        num_filters : int
                The number of filters to use in SchNet (default : `128`)
        num_interactions : int
                The number of interaction blocks in SchNet (default : `6`)
        num_gaussians : int
                The number of gaussians :math:`\mu` in SchNet (default : `50`).
        cutoff : float
                The cutoff distance for interatomic interactions in SchNet (default : `10.0`).
        atomref : torch.Tensor
                The reference of single-atom properties.
        input_size : int
                
        hidden_size : int

        output_size : int


        """
        super(VAMPSchNet, self).__init__()


        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = ModuleList()

        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, hidden_channels) # JG: change output vector size

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

        self.vamp_lin1 = Linear(input_size, hidden_size)
        self.vamp_lin2 = Linear(hidden_size, hidden_size)
        self.vamp_lin3 = Linear(hidden_size, hidden_size)
        self.vamp_lin4 = Linear(hidden_size, hidden_size)
        self.vamp_lin5 = Linear(hidden_size, hidden_size)
        self.vamp_lin6 = Linear(hidden_size, output_size)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
           self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, data, batch=None):
        #print("data entering SchNet...")
        pos = data # torch.Size([10]) 
        
        # featurization via SchNet for each of all datapoints in the batch
        #print("shape of pos: ", pos.size())

        # change batch_size here:
        num_nodes = len(atom_indices)
        batch_size = int(pos.size(0) / num_nodes)# takes either a batch or all datapoints
        #print("batch_size: ", batch_size)

        # get the node feature matrix in a tensor and pass onto deivice 
        # TODO: more sophisticated atom typing
        z = torch.LongTensor(atom_type * batch_size) # len(atom_type) = number of features to include
        #print("(before) total atom type: ", z)
        #print("(before) shape total atom type: ", z.shape)
        #print("(before) data type: ", z.dtype)
        z = z.to(device=device, non_blocking=True, dtype=torch.long) # JG
        assert z.dim() == 1 and z.dtype == torch.long

        #batch = torch.zeros_like(z) if batch is None else batch
        batch = torch.arange(batch_size).view(-1,1).repeat(1,num_nodes).view(-1)
        batch = batch.to(device=device, non_blocking=True, dtype=torch.long)
        #print("(after) total atom type: ", z)
        #print("(after) shape total atom type: ", z.shape)
        #print("(after) data type: ", z.dtype)
        #print("embedding weights: ")
        #print(self.embedding.weight.data) # JG

        h = self.embedding(z) # JG
        #print("h dtype: ", h.dtype)
        #print("h[0][:]: ", h[0][:])
        #print("h[1][:]: ", h[1][:])
      

        #print("initial embedding: ", h)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            #print("shape of embedding: ", h.size())
            #print("shape of edge_index: ", edge_index.size())
            #print("shape of edge_weight: ", edge_weight.size())
            #print("shape of edge_attr: ", edge_attr.size())
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        #print("h dtype after interaction: ", h.dtype)
        #print("h[0][:] after interaction: ", h[0][:])
        #print("h[1][:] after interaction: ", h[1][:])
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        x = global_mean_pool(h, batch, batch_size)# torch.Size([16, hidden_channels])
        #print("featurization: ", x)
        #print('mean size: ', x.shape)
        #x = h.view(batch_size, num_nodes*h.size(-1))
        #print("data type: ", h.dtype)
        #print("data size: ", x.size())

        # pass the conformation embedding to VAMPnet
        #print("data entering VAMPnet...")
        x = F.elu(self.vamp_lin1(x)) # x: torch.Size([batch_size, embedding_size]
        x = F.elu(self.vamp_lin2(x))
        x = F.elu(self.vamp_lin3(x))
        x = F.elu(self.vamp_lin4(x))
        x = F.elu(self.vamp_lin5(x))
        x = F.softmax(self.vamp_lin6(x), dim=1)
        #print("softmax:")
        #print(x)
        return x

# define the training function
# change batch_size here:
def train_with_sampler(n_features, n_epochs, batch_size=64, learning_rate=1e-4):
    # define shared hyperparameters
    num_nodes = n_features # number of atoms of interest

    # define hyperparameters for SchNet
    hidden_channels = 32 # change embedding size here
    num_filters = 32
    num_interactions = 6
    num_gaussians = 50
    cutoff = 10.0
    atomref = None


    # define hyperparameters for VAMPnet
    input_size = hidden_channels
    hidden_size = 32
    output_size = 8

    lobe = VAMPSchNet(hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, atomref, input_size, hidden_size, output_size)
    lobe = lobe.to(device)

    vschnet = dt.decomposition.VAMPNet(lobe, device=device, optimizer='Adam', learning_rate=learning_rate)

    from torch.utils.data import RandomSampler, BatchSampler

    sampler = BatchSampler(RandomSampler(total_train_data, True), batch_size=batch_size, drop_last=True)
    loader = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*total_train_data[:]),
                                         drop_last=True, batch_size=1, sampler=sampler)


    sampler_val = BatchSampler(RandomSampler(total_val_data, True), batch_size=batch_size, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*total_val_data[:]),
                                         drop_last=True, batch_size=1, sampler=sampler_val)

    val_scores = list()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}...")
        #print(f"Epoch {epoch}...")
        lobe.train()
        #count = 0 # JG
        for batch_0, batch_t in loader:
            #print(f"batch {count}...") # JG
            # here the shape of batch_0 and batch_t are both: torch.Size([1, batch_size, 10, 3]) 
            batch_0 = batch_0.view(1, batch_size*num_nodes, 3)
            batch_t = batch_t.view(1, batch_size*num_nodes, 3)
            #print("shape of batch_0: ", batch_0.size())
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            vschnet.partial_fit((batch_0, batch_t)) # original
            #count += 1 # JG
        lobe.eval()
        scores_val = []
        # TODO: employ pytorch_geometric DataLoader
        for batch_0, batch_t in loader_val:
            # here the shape of batch_0 and batch_t are both: torch.Size([1, batch_size, num_nodes, 3]) 
            batch_0 = batch_0.view(1, batch_size*num_nodes, 3)
            batch_t = batch_t.view(1, batch_size*num_nodes, 3)
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            score_val = vschnet.validate((batch_0, batch_t))
            scores_val.append(score_val.cpu().numpy())
        print(f"Epoch {epoch+1}/{n_epochs}: Validation score {np.mean(scores_val):.4f}", end='\r')
        val_scores.append(np.mean(scores_val)) # record mean validation score from each epoch
    return vschnet, val_scores
vschnet, val_scores = train_with_sampler(num_nodes, 100, learning_rate=5e-4)  # or the easier version, train(...)

train_scores = vschnet.train_scores.T # numpy array with [[index] [score]]
val_scores = np.array([np.array(np.arange(len(val_scores))), np.array(val_scores)])

# output the training and validation scores into npy files
np.save('train_scores.npy', train_scores)
np.save('val_scores.npy', val_scores)

# analysis #1: state probabilities for each of the output states
# final results: a list containing three trajectories, each contains fuzzy membership of each point into 8 states [np.ndarray (31, 4000, 3)]
# process each trajectory X in coordinates and save results separately
final_results = list()

for index in traj_indices:
    print(f"traj index: {index}")
    traj = md.load(f'../data/run{index[0]}-clone{index[1]}.h5')
    topology = traj.topology
    pos = traj.xyz[:,atom_indices,:] # pos is np.array of shape (traj_len, num_nodes, 3)
    #new_batch_size = len(traj)
    new_batch_size = 32
    new_loader = torch.utils.data.DataLoader(pos, new_batch_size, shuffle=False)
    results = list()
    for data in new_loader:
        # reshape traj data to (n_data*n_nodes)*n_coord
        data = data.reshape((new_batch_size*num_nodes, 3))
        results.append(vschnet.transform(data))
    final_results.append(np.concatenate((results), axis=0))
np.save(f'transformed_trajs.npy', final_results)

