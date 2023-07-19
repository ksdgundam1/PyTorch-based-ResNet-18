import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available else 'cpu'


batch_size = 10
learning_rate = 1e-3
epochs = 100


import ssl

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

#download 
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = True)

test_set = torchvision.datasets.CIFAR10(root = '/data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, drop_last = True)


class myResnet(nn.Module):
    def __init__(self):
        super(myResnet, self).__init__()
        
        self.blockA = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.blockB = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.blockC = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.blockD = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        self.blockE = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )

        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.fc = nn.Linear(8 * 8 * 1024, 10) 
        
        #skip connections
        self.conv1X1_a = nn.Conv2d(64, 128, kernel_size = 1, stride = 1)
        self.conv1X1_b = nn.Conv2d(128, 256, kernel_size = 1, stride = 1)
        self.conv1X1_c = nn.Conv2d(256, 512, kernel_size = 1, stride = 1)
        self.conv1X1_d = nn.Conv2d(512, 1024, kernel_size = 1, stride = 1)
        
    
    def forward(self, x):
        #layer 1
        out = self.blockA(x)
        
        #layer2
        skip = self.conv1X1_a(out)
        out = self.blockB(out)
        #out = out + skip
        
        #layer3
        skip = self.conv1X1_b(out)
        out = self.blockC(out)
        #out = out + skip
        
        #layer 4
        skip = self.conv1X1_c(out)
        out = self.blockD(out)
        #out = out + skip
        
        #layer 5
        skip = self.conv1X1_d(out)
        out = self.blockE(out)
        #out = out + skip
        
        #fc layer
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


model = myResnet()
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)


#train & evaluate
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #move data to device
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

      
        running_loss += loss.item()
        if i % 200 == 199:  
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (100 * correct / total))

