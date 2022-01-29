import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import datetime

args = {
        'GPU_NUM' : 2,
        'Epochs' : 200,
        'batch_size' : 128,
        'lr' : 0.0002,
        'b1' : 0.5,
        'b2' : 0.999,
        'latent_dim' : 62,
        'code_dim' : 2,
        'n_classes' : 2,
        'img_size' : 32,
        'channels' : 1,
        'sample_interval' : 400
        }

device = torch.device('cuda:{}'.format(args['GPU_NUM']) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
today = datetime.date.today()

my_transform =transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(args['img_size']), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5], [0.5])])

train_data = ImageFolder('MNIST/classes/binary/train', transform = my_transform)
test_data = ImageFolder('MNIST/classes/binary/test', transform = my_transform)

train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        #nn.Conv2d(input channel, output channel, ...)
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size = 3,
                               stride=stride,
                               padding=1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(planes) # Batchnorm은 사이의 가중치가 아니라 출력 층만 노말라이징
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size = 3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size = 1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes))
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet_3232(nn.Module):
    def __init__(self, channels = 1, num_classes = 10):
        super(ResNet_3232, self).__init__()
        
        self.rgb = channels
        self.in_planes = 16
        # RGB여서 3, in_planes는 내맘대로 16
        self.conv1 = nn.Conv2d(self.rgb,16,
                               kernel_size = 3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64,2)
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] *(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes,planes,stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = F.avg_pool2d(out4, 4)
        out6 = out5.view(out5.size(0), -1)
        out7 = self.linear(out6)
        
        return out7

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval ==0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
    return output

def evaluate(model, test_loader):
    model.eval()
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1,keepdim=True)[1] # output에서 제일 큰 놈의 index를 반환한다(이경우에 0 or 1)
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct / len(test_loader.dataset)
    return test_loss, test_accuracy

if __name__ == '__main__':
    print('오늘 날짜 :',today)
    print('GPU device :', device)
    # model = ResNet_3232(channels = args['channels'], num_classes=args['n_classes']).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # best_accuracy = 0
    # for Epoch in range(1, 10+1):
    #     train(model, train_loader, optimizer, log_interval=200)
    #     test_loss, test_accuracy = evaluate(model, test_loader)
    #     if test_accuracy > best_accuracy:
    #         best_accuracy = test_accuracy
    #         torch.save(model, 'pretrained_model/ResNet_3232_1_7.pt')
    #         torch.save(model.state_dict(), 'pretrained_model/ResNet_3232_parameters_1_7.pt')
    #     print("[EPOCH: {}], \tTest Loss: {:.4f},\tTest Accuracy: {:.2f}%\n".format(
    #         Epoch, test_loss, test_accuracy))