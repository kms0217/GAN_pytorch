import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets as dset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import imageio
import glob

# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device use : {device}")

# seed
def seed_everything(seed: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    
seed_everything()

# Generator
# input : 128 noise, output = MNIST.shape
class Generator(nn.Module):
    def __init__(self, input_dim = 128, hidden_dims = [256, 512, 1024], original_img_size = [28, 28], image_channel = 1, norm = False):
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        self.img_size = original_img_size
        self.img_channel = image_channel
        self.out_dim = np.prod(self.img_size) * image_channel
        layers = []
        pre_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(pre_dim, hdim))
            if norm :
                layers.append(nn.BatchNorm1d(hdim), 0.8)
            layers.append(nn.ReLU(True))
            pre_dim = hdim
        layers.append(nn.Linear(pre_dim, self.out_dim))
        layers.append(nn.Tanh())
        
        for idx, layer in enumerate(layers):
            layer_name = f"{type(layer).__name__.lower()}_{idx}"
            self.net.add_module(layer_name, layer)
    
    def forward(self, x):
        image = self.net(x)
        image = image.view(image.shape[0], self.img_channel, self.img_size[0], self.img_size[1])
        return image

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim = 28 * 28, hidden_dims = [1024, 512, 256, 128], output_dim = 1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential()
        layers = []
        pre_dim = input_dim
        for hdim in hidden_dims :
            layers.append(nn.Linear(pre_dim, hdim))
            layers.append(nn.ReLU(True))
            pre_dim = hdim
        layers.append(nn.Linear(pre_dim,output_dim))
        layers.append(nn.Sigmoid())
        for idx, layer in enumerate(layers):
            layer_name = f"{type(layer).__name__.lower()}_{idx}"
            self.net.add_module(layer_name, layer)
                
    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))

def show_image(images, epoch, plt_show = False):
    if not os.path.exists("generated_samples"):
        os.mkdir("generated_samples")
    fig = plt.figure(figsize = (8, 8))
    columns = 10
    rows = 10
    for i in range(1, images.shape[0] + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1].permute(1, 2, 0), cmap = "gray", vmin = -1, vmax = 1)
    plt.savefig(f'./generated_samples/sampleepoch{epoch:03}.png')
    if plt_show :
        plt.show()

mnist_dataset = dset.MNIST(root = "data/MNIST", 
                          train = True, 
                          transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]), 
                          download = True)

batch_size = 100                          
data_iter = DataLoader(mnist_dataset, batch_size = batch_size, shuffle = True)
# fixed sample noise for checking train generator
sample_noise = torch.FloatTensor(np.random.randn(100, 128)).to(device)
learning_rate = 1e-4
epochs = 100

critrion = nn.BCELoss()
G = Generator().to(device)
D = Discriminator().to(device)
G_optim = optim.Adam(G.parameters(), lr = learning_rate)
D_optim = optim.Adam(D.parameters(), lr = learning_rate)

for epoch in range(epochs):
    D_loss_sum = 0
    G_loss_sum = 0

    for i, (image, _) in enumerate(data_iter):
        image = image.to(device)
        noise = torch.FloatTensor(np.random.randn(image.shape[0], 128)).to(device)
        # ground truth
        real = torch.FloatTensor(np.ones((image.shape[0], 1))).to(device)
        fake = torch.zeros_like(real)

        # Discriminator train
        G.eval()
        D.train()
        R_loss = critrion(D(image), real)
        D_optim.zero_grad()
        R_loss.backward()
        D_optim.step()
        
        F_loss = critrion(D(G(noise).detach()), fake)
        F_loss.backward()
        D_optim.step()
        
        D_loss = R_loss + F_loss
        
        # Generator train
        G.train()
        D.eval()
        
        G_out = G(noise)
        G_loss = critrion(D(G_out), real)
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
        
        D_loss_sum += D_loss.item()
        G_loss_sum += G_loss.item()
    print(f"epoch : {epoch :02} D_loss : {D_loss_sum / len(data_iter) :.5} G_loss : {G_loss_sum / len(data_iter) : .5}")
    
    # save sample image
    G.eval()
    with torch.no_grad():
        sample_out = G(sample_noise)
        show_image(sample_out.cpu(), epoch + 1)
    G.train()
    
# make gif
anim_file = './gan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./generated_samples/sample*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
print("finish")