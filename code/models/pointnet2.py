#This comes from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

import os.path as osp
import os
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, DynamicEdgeConv,fps, radius, global_max_pool


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        if self.ratio != 1:
            idx = fps(pos, batch, ratio=self.ratio)
        else:
            idx = torch.arange(pos.size(0))

        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self,num_classes,input_channels):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([input_channels+3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, x,pos,batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

class EdgeNet(torch.nn.Module):
    def __init__(self,num_classes,input_channels,chan=64,k=5,agregFunc=global_max_pool):
        super(EdgeNet, self).__init__()

        self.lin1 = MLP([2*(input_channels+3), chan])
        self.dynEdgeConv1 = DynamicEdgeConv(self.lin1,k=k)

        self.lin2 = MLP([2*chan, chan])
        self.dynEdgeConv2 = DynamicEdgeConv(self.lin2,k=k)

        self.lin3 = MLP([2*chan, chan*2])
        self.dynEdgeConv3 = DynamicEdgeConv(self.lin3,k=k)

        self.lin4 = MLP([2*chan*2, chan*4])
        self.dynEdgeConv4 = DynamicEdgeConv(self.lin4,k=k)

        self.lin5 = MLP([chan*8, chan*16])

        self.agregFunc = agregFunc

        self.finalMLP =  MLP([chan*16,chan*8,chan*4])

        self.finalLin = torch.nn.Linear(chan*4,num_classes)

    def forward(self,x,pos,batch):

        if x is None:
            x = pos
        else:
            x = torch.cat((x,pos),dim=-1)

        #in_chan
        x1 = self.dynEdgeConv1(x,batch)
        #chan
        x2 = self.dynEdgeConv2(x1,batch)
        #chan
        x3 = self.dynEdgeConv3(x2,batch)
        #2*chan
        x4 = self.dynEdgeConv4(x3,batch)
        #4*chan
        x = torch.cat((x1,x2,x3,x4),dim=-1)
        #8*chan
        x = self.lin5(x)
        #16*chan
        x = self.agregFunc(x,batch)
        #16*chan
        x = self.finalMLP(x)
        #4*chan
        x = self.finalLin(x)
        #num_classes

        return x


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data.x,data.pos,data.batch), data.y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "../models/pn_pretraining/modelPN_epoch{}".format(epoch))

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x,data.pos,data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    if not os.path.exists("../models/pn_pretraining"):
        os.makedirs("../models/pn_pretraining")

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(10,0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    last_test_acc = None
    bestEpoch = 0

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
        if last_test_acc is None:
            last_test_acc = test_acc

        if test_acc > last_test_acc:
            torch.save(model.state_dict(), "../models/pn_pretraining/modelPN_best_epoch{}".format(epoch))
            if os.path.exists("../models/pn_pretraining/modelPN_best_epoch{}".format(bestEpoch)):
                os.remove("../models/pn_pretraining/modelPN_best_epoch{}".format(bestEpoch))
            bestEpoch = epoch

        last_test_acc = test_acc
