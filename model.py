from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
import torch
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]


def init_embeddings(x):
    x = x.weight.data
    value = 2 / (x.size(1) + 1)
    x.uniform_(-value, value)
    
class StructuredData(object):
    def __init__(self, df, y, cat_flds, cont_flds, val_index=None, batch_size=32, shuffle=False, num_workers=1):
        self.val_index = val_index
        if val_index:
            ((val_df, df), (y_val, y)) = split_by_idx(val_index, df, y)

        train_dataset = StructuredDataSet(df[cat_flds], df[cont_flds], y)
        
        if shuffle:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                       sampler=train_sampler, num_workers=num_workers)
        
        if val_index:
            validation_dataset = StructuredDataSet(val_df[cat_flds], val_df[cont_flds], y_val)
            self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                                            num_workers=num_workers)
        else:
            self.validation_loader = None
            
    def get_data(self):  
        if self.val_index:
            return self.train_loader, self.validation_loader
        else:
            return self.train_loader  
    
class StructuredDataSet(Dataset):
    def __init__(self, cats, conts, y):
        self.cats = np.asarray(cats, dtype=np.int64)
        self.conts = np.asarray(conts, dtype=np.float32)
        self.N = len(y)
        y = np.zeros((n,1)) if y is None else y[:,None]
        self.y = np.asarray(y, dtype=np.float32)
            
    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]
    

     
class MultiInputNN(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, sizes, drops, y_range=None, use_bn=False, f=F.relu):
        super().__init__()
        # embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(insize, outsize) for insize, outsize in emb_szs])
        for layer in self.embeddings:
            init_embeddings(layer)
        self.num_categorical = sum([layer.embedding_dim for layer in self.embeddings])

        self.num_numerical = n_cont
        # linear layers
        sizes = [self.num_categorical + self.num_numerical] + sizes
        self.linear = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(size) for size in sizes[1:]])
        for layer in self.linear:
            kaiming_normal(layer.weight.data)
        # dropout layers
        
        self.emb_drop = nn.Dropout(emb_drop)
        self.drop_out = [nn.Dropout(drop) for drop in drops]
        # output layer
        self.output = nn.Linear(sizes[-1], 1)
        kaiming_normal(self.output.weight.data)
        self.f = f
        self.bn = nn.BatchNorm1d(self.num_numerical)
        self.use_bn = use_bn
        self.y_range = y_range

    def forward(self, x_cat, x_cont):
        if self.num_categorical > 0:
            X = [emb_layer(x_cat[:,i]) for i, emb_layer in enumerate(self.embeddings)]
            X = torch.cat(X, 1)
            X = self.emb_drop(X)
        if self.num_numerical > 0:
            X2 = self.bn(x_cont)
            X = torch.cat([X, X2], 1) if self.num_categorical != 0 else X2
        for linear, drop, norm in zip(self.linear, self.drop_out, self.bns):
            X = self.f(linear(X))
            if self.use_bn: 
                X = norm(X)
            X = drop(X)
        X = self.output(X)
        if self.y_range:
            X = F.sigmoid(X)
            X = X * (self.y_range[1] - self.y_range[0])
            X = X + self.y_range[0]
        return X
    

    
    def fit(self, train_loader, learning_rate=1e-3, batch_size=64, epochs=1, save=False, save_path='tmp/checkpoint.pth.tar', 
                    pre_saved=False):

        loss = nn.MSELoss()
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
        n_batches = int(train_loader.dataset.N / train_loader.batch_size)

        if pre_saved:
            checkpoint = torch.load(save_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('...restoring model...')

        for epoch in range(epochs):
            if pre_saved:
    #             print('Here', 'start_epoch', start_epoch, 'epoch', epoch)
    #             epoch = start_epoch
                pre_saved = False

            epoch_loss = 0.
            for i, (batch_cat, batch_cont, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_cat, batch_cont, batch_y = Variable(batch_cat), Variable(batch_cont), Variable(batch_y)

                y_hat = model.forward(batch_cat, batch_cont)
                l = loss(y_hat, batch_y)
                epoch_loss += l.data[0]


                l.backward()
                optimizer.step()
                if i != 0 and i % 2000 == 0:
                    acc = rmse(y_hat.data.numpy(), batch_y.data.numpy())
                    print('iteration: {} of n_batches: {}'.format(i, n_batches))
            
            print('epoch: {}, train_loss: {}'.format(epoch, epoch_loss / n_batches))

        if save:
            state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
            self.save_checkpoint(state, filename=save_path)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        
    def my_predict(self, df, cat_flds, cont_flds):
        self.eval()
        cats = np.asarray(df[cat_flds], dtype=np.int64)
        conts = np.asarray(df[cont_flds], dtype=np.float32)
        x_cat = Variable(torch.from_numpy(cats))
        x_cont = Variable(torch.from_numpy(conts))
        pred = self.forward(x_cat, x_cont)
        return pred.data.numpy().flatten()
