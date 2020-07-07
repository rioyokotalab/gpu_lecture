import torch

epochs = 300
batch_size = 32
D_in = 784
H = 100
D_out = 10
learning_rate = 1.0e-04

# create random input and output data
x = torch.randn(batch_size, D_in)
y = torch.randn(batch_size, D_out)

# define model
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

# define loss function
criterion = torch.nn.MSELoss(reduction='sum')

for epoch in range(epochs):
    # forward pass: compute predicted y
    y_p = model(x)

    # compute and print loss
    loss = criterion(y_p, y)
    print(epoch, loss.item())

    # backward pass
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        # update weights
        for param in model.parameters():
            param -= learning_rate * param.grad
