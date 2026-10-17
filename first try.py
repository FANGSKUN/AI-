import torch
import torch.nn as nn
n,seed=100,1
def get_xy():
    X = torch.rand(n,1)
    Y = 5*X + 4 + torch.randn(n,1)*0.1
    return X,Y
epochs=10000
def train_model(X,Y,epochs):
    model = nn.Linear(1,1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pre = model(X)
        loss=loss_fn(model(X),Y)
        loss.backward()
        optimizer.step()
    return model

def main():
    X,Y = get_xy()
    mode = train_model(X,Y,epochs)
    w=mode.weight.item()
    b=mode.bias.item()
    print(w,b)

if __name__ == '__main__':
    main()