import unidecode
import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#Generate pseudorandom data
def generate_data(num_samples,p1,markovity):
    p0 = 1 - p1
    data = np.empty([num_samples,1],dtype=np.uint8)
    print(data.shape)
    data[:markovity] = np.random.choice([65, 66], size=(markovity,1), p=[p0, p1])
    for i in range(markovity, num_samples):
        if data[i-1] == data[i-markovity]:
            data[i] = 65
        else:
            data[i] = 66
    return data

#reads in a file using unicode symbols (eg: a text file)
def read_unicode_file(file_name):
    data = unidecode.unidecode(open(file_name).read())
    return data

#outputs a training batch
def get_training_batch(batch,n_batches,data):
    data_len = len(data)
    chunk_size = int(data_len/n_batches)
    if batch == 0:
        chunk_start = batch*chunk_size
    else:
        chunk_start = batch*chunk_size - 1
    
    chunk = data[chunk_start:(batch+1)*chunk_size]
    inp = char_to_tensor(chunk[:-1])
    target = char_to_tensor(chunk[1:])
    return inp, target

# Turn string into a list of longs
def char_to_tensor(inp):
    all_characters = string.printable
    tensor = torch.zeros(len(inp)).long()
    for c in range(len(inp)):
        tensor[c] = all_characters.index(inp[c])
    return Variable(tensor)

#The simple RNN (GRU) model, where the state is reused
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input_1 = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input_1.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


def train_step(inp, target, model, criterion, optimizer, hidden=None):
    if hidden is None:
        hidden = model.init_hidden()
    model.zero_grad()
    loss = 0
    
    chunk_len = len(target)
    for c in range(chunk_len):
        output, hidden = model(inp[c], hidden)
        loss += criterion(output, target[c].view(1))

    loss.backward()
    optimizer.step()
    return loss.data.item()/(chunk_len*np.log(2.0)), hidden.data

def train_model(data):
    n_epochs = 1
    n_batches = 200000
    print_every = 100
    hidden_size = 64
    n_layers = 1
    lr = 0.003
    n_characters = 256

    rnn_model = RNN(n_characters, hidden_size, n_characters, n_layers)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
    loss_criterion = nn.CrossEntropyLoss()

    start = time.time()
    all_losses = []
    loss_avg = 0
    hidden=None

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            inp,target = get_training_batch(batch,n_batches,data)
            loss,hidden = train_step(inp,target,rnn_model,loss_criterion,rnn_optimizer,hidden)       
            loss_avg += loss

            if batch % print_every == 0:
                print('[%s (%d, %d, %d%%) %.4f]' % (time_since(start), epoch, batch, (batch+ epoch*n_batches)/(n_epochs*n_batches)* 100, loss))
                #print(evaluate('Wh', 100), '\n')
                print('avg_loss: \n', loss_avg / print_every)
                all_losses.append(loss_avg / print_every)
                loss_avg = 0


def main():
    n_samples=10000000
    file_name="data/input.txt"
    data = generate_data(n_samples,0.5,30);
    np.savetxt(file_name,data,delimiter='', fmt='%c',newline='');

    data = read_unicode_file(file_name)
    train_model(data);


if __name__ == '__main__':
    main()
