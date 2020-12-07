import numpy as np
from math import pi as PI
import ase
import torch
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import torch.nn.functional as F
from torch_geometric.nn import SchNet, radius_graph, MessagePassing
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

# get the number of datapoints
data_len = concat_coords.shape[0]

# reshape the coordinates
pos = concat_coords.reshape((data_len, 10, 3)) # pos is a np.array
print(pos.shape, concat_dih.shape)


# separate the data into training and validation
n_train = int(np.floor(len(pos) * train_ratio))
n_validation = len(pos) - tau - n_train

# separate the time-lagged datapoints and shuffle data
dataset = dt.data.TimeLaggedDataset.from_trajectory(tau, pos.astype(np.float32))
train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_validation])

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
        num_nodes = 10
        batch_size = int(pos.size(0) / num_nodes)# takes either a batch or all datapoints

        # get the node feature matrix in a tensor and pass onto deivice 
        # TODO: more sophisticated atom typing
        z = torch.LongTensor([0, 0, 1, 2, 0, 0, 0, 1, 2, 0] * batch_size)
        z = z.to(device=device, non_blocking=True, dtype=torch.long)
        assert z.dim() == 1 and z.dtype == torch.long

        #batch = torch.zeros_like(z) if batch is None else batch
        batch = torch.arange(batch_size).view(-1,1).repeat(1,num_nodes).view(-1)
        batch = batch.to(device=device, non_blocking=True, dtype=torch.long)

        h = self.embedding(z)
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

        h = self.lin1(h) 
        h = self.act(h)
        h = self.lin2(h)
        x = h.view(batch_size, num_nodes*h.size(-1))
        #print("data type: ", h.dtype)
        #print("data size: ", h.size())

        # pass the conformation embedding to VAMPnet
        #print("data entering VAMPnet...")
        x = F.elu(self.vamp_lin1(x)) # x: torch.Size([batch_size, 30]
        x = F.elu(self.vamp_lin2(x))
        x = F.elu(self.vamp_lin3(x))
        x = F.elu(self.vamp_lin4(x))
        x = F.elu(self.vamp_lin5(x))
        x = F.softmax(self.vamp_lin6(x), dim=1)

        return x




# define the training function
# change batch_size here:
def train_with_sampler(n_epochs, batch_size=2048, learning_rate=1e-4):
    # define shared hyperparameters
    num_nodes = 10

    # define hyperparameters for SchNet
    hidden_channels = 128 # change embedding size here
    num_filters = 128
    num_interactions = 6 
    num_gaussians = 50
    cutoff = 10.0
    atomref = None

    
    # define hyperparameters for VAMPnet
    input_size = num_nodes * hidden_channels
    hidden_size = 128
    output_size = 6
 
    lobe = VAMPSchNet(hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, atomref, input_size, hidden_size, output_size)
    lobe = lobe.to(device)    
    
    vschnet = dt.decomposition.VAMPNet(lobe, device=device, optimizer='Adam', learning_rate=learning_rate)

    from torch.utils.data import RandomSampler, BatchSampler    

    sampler = BatchSampler(RandomSampler(train_data, True), batch_size=batch_size, drop_last=True)
    loader = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*train_data[:]), 
                                         drop_last=True, batch_size=1, sampler=sampler)

    sampler_val = BatchSampler(RandomSampler(val_data, True), batch_size=batch_size, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dt.data.TimeLaggedDataset(*val_data[:]), 
                                         drop_last=True, batch_size=1, sampler=sampler_val)
    
    '''
    # get the node feature matrix in a tensor and pass onto deivice 
    # TODO: more sophisticated atom typing
    z = torch.LongTensor([0, 0, 1, 2, 0, 0, 0, 1, 2, 0] * batch_size)
    z = z.to(device=device, non_blocking=True, dtype=torch.long)
    '''

    val_scores = list()
    for epoch in range(n_epochs):
        #print(f"Epoch {epoch+1}/{n_epochs}...")

        lobe.train()       
        for batch_0, batch_t in loader:
            # here the shape of batch_0 and batch_t are both: torch.Size([1, batch_size, 10, 3]) 
            batch_0 = batch_0.view(1, batch_size*10, 3)
            batch_t = batch_t.view(1, batch_size*10, 3)
            #print("shape of batch_0: ", batch_0.size())
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            vschnet.partial_fit((batch_0, batch_t)) # original

        lobe.eval()
        scores_val = []
        # TODO: employ pytorch_geometric DataLoader
        for batch_0, batch_t in loader_val:
            # here the shape of batch_0 and batch_t are both: torch.Size([1, batch_size, 10, 3]) 
            batch_0 = batch_0.view(1, batch_size*10, 3)
            batch_t = batch_t.view(1, batch_size*10, 3)
            batch_0 = batch_0[0].to(device=device, non_blocking=True, dtype=torch.float32)
            batch_t = batch_t[0].to(device=device, non_blocking=True, dtype=torch.float32)
            score_val = vschnet.validate((batch_0, batch_t))
            scores_val.append(score_val.cpu().numpy())
        print(f"Epoch {epoch+1}/{n_epochs}: Validation score {np.mean(scores_val):.4f}", end='\r')
        val_scores.append(np.mean(scores_val)) # record mean validation score from each epoch
    return vschnet, val_scores
vschnet, val_scores = train_with_sampler(100, learning_rate=5e-4)  # or the easier version, train(...)

train_scores = vschnet.train_scores.T # numpy array with [[index] [score]]
val_scores = np.array([np.array(np.arange(len(val_scores))), np.array(val_scores)])

# output the training and validation scores into npy files
np.save('train_scores.npy', train_scores)
np.save('val_scores.npy', val_scores)

# analysis #1: state probabilities for each of the output states
# final results: a list containing three trajectories, each contains fuzzy membership of each point into 6 states [np.ndarray (250000, 6)]

# process each trajectory X in coordinates and save results separately
final_results = list()
count = 0
for X in coordinates:
    new_loader = torch.utils.data.DataLoader(X, batch_size=25000, shuffle=False)
    print("count: ", count)
    results = list()
    for data in new_loader:
        # reshape X from n_data*(n_node*n_coord) to (n_data*n_node)*n_coord for pytorch_geometric
        data = data.reshape((250000, 3))
        results.append(vschnet.transform(data))
    final_results.append(np.concatenate((results), axis=0))
    count += 1
np.save(f'transformed_trajs.npy', final_results)
