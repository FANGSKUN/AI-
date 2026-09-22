import numpy as np

def make_data():
    np.random.seed(2)
    X = np.random.rand(100,1)
    noise = np.random.randn(100,1)*0.2
    Y = 4*X + noise + 5
    return X,Y

def pred(X,w,b):
    return X @ w + b

def loss(X,Y,w,b):
    loss = np.mean((pred(X,w,b)-Y)**2)
    return loss

def grad(X,Y,w,b):
    lose = pred(X,w,b) - Y
    dw = 2*np.mean(lose*X)
    db = 2*np.mean(lose)
    return dw,db


if __name__ == "__main__":
    lr,epoch = 0.01,40000
    X,Y = make_data()
    w = np.zeros((1, 1))
    b = 0.0
    for i in range(epoch):
        dw,db = grad(X,Y,w,b)
        w = w - lr*dw
        b = b - lr*db

    print(w,b)