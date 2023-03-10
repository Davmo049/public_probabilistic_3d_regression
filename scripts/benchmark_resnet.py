import torch
import torchvision.models.resnet as resnet
import time

def main():
    device = torch.device('cuda')
    print('resnet18')
    net = resnet.resnet18(num_classes=17*7).to(device)
    opt = torch.optim.Adam(net.parameters())
    batch_data = torch.zeros((32,3,224,224), dtype=torch.float32, device=device)
    T = 1000
    start = time.time()
    for _ in range(T):
        out = net(batch_data)
        loss = torch.sum(out.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    end = time.time()
    print((end-start)/T)
    print('resnet50')
    net =resnet.resnet50(num_classes=17*7).to(device)
    opt = torch.optim.Adam(net.parameters())
    start = time.time()
    for _ in range(T):
        out = net(batch_data)
        loss = torch.sum(out.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    end = time.time()
    print((end-start)/T)
    print('resnet101')
    net = resnet.resnet101(num_classes=17*7).to(device)
    opt = torch.optim.Adam(net.parameters())
    start = time.time()
    for _ in range(T):
        out = net(batch_data)
        loss = torch.sum(out.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    end = time.time()
    print((end-start)/T)

if __name__ == '__main__':
    main()
