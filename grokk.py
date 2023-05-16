import pathlib
import pickle
import torch
import numpy as np
import scipy.optimize
from scipy.sparse.linalg import eigsh, LinearOperator
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import copy

import matplotlib.pylab as plt
import matplotlib.pyplot as plts
import matplotlib.gridspec as gridspec

# Vision
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset

dtype = torch.float32

def make_data(p, N_c, alpha, device, seed):
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.seed(seed)

    random.shuffle(pairs)
    X = torch.tensor(pairs)
    X_one_hot = F.one_hot(X, num_classes=N_c).to(device=device, dtype=dtype)

    #Y = ( 2*X[:,1] * X[:,0] + 1*X[:,0]**2 +  1*X[:,1]**2  ) % p
    #Y = ( X[:,0]**3 + X[:,0] * (X[:,1])**2 + X[:,1]) % p
    Y = (X[:, 0]**1 + X[:, 1]**1) % p
    m = Y.shape[0]

    Labels = F.one_hot(Y, num_classes=p).to(device=device, dtype=dtype)

    X_one_hot = X_one_hot.to(device)
    Y = Y.to(device)
    Labels = Labels.to(device)

    ### Loader ###

    train_size = int(alpha * m)

    test_size = m - train_size  # between 1 and m


    # here is the updated test and train, we restrict to full batch, but it's easy to
    # keep everything on the GPU
    loader_train = [(X_one_hot[:train_size], Y[:train_size], Labels[:train_size])]
    loader_test = [(X_one_hot[train_size:], Y[train_size:], Labels[train_size:])]

    return loader_test, loader_train



def test_loss(model, loader_test):
    criterion1 = nn.MSELoss()
    loss = 0
    for x, y, y_one_hot in loader_test:
        model.eval()  # put model to training mode
        # model = model.to(device=device)
        # no longer need to move to device
        # x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        # y = y.to(device=device, dtype=torch.long)
        scores = model(x).squeeze()
        loss += criterion1(scores, y_one_hot)
        #loss += criterion2(scores.squeeze(), y)
    return loss


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    x_wrong = []
    with torch.no_grad():
        for x, y, y_one_hot in loader:

            # x = x.to(device=device, dtype=dtype)
            # y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            x_wrong.append(x[y != preds])

        acc = float(num_correct) / num_samples

    return num_correct.data, num_samples, acc


def hessian_trace(model, loss_fn, x, y):
    # compute the gradient of the loss with respect to the model parameters
    # via an approximation
    params = list(model.parameters())
    loss = loss_fn(model(x), y)
    grad_params = torch.autograd.grad(loss, params, create_graph=True)

    # compute the trace of the Hessian
    hv = 0
    v_norm = 0.0
    for i, (param, grad) in enumerate(zip(params, grad_params)):
        grad_flat = grad.view(-1)
        v = torch.randn_like(grad_flat)
        grad_dot_v = grad_flat.dot(v)
        # here we don't retain the graph on the last iteration of the loop
        grad_hv = torch.autograd.grad(
            grad_dot_v, [param], retain_graph=(i != len(grad_params) - 1))
        hv += grad_hv[0].view(-1).dot(v).item()
        v_norm += torch.sum(torch.square(v)).item()

    trace_approx = hv / v_norm
    return trace_approx


def hessian_vector_product_function(model, loss_fn, x, y, device, params):
    # returns a function which computes the the product of the hessian of the loss
    # on the model with a vector (on the train set)
    def calc_hvp(vec):
        loss = loss_fn(model(x), y)
        vec = torch.tensor(vec, dtype=torch.float32,
                           requires_grad=False, device=device)
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        grad_params_flat = torch.cat([g.flatten() for g in grad_params])
        grad_dot_vec = torch.dot(grad_params_flat, vec)
        hvp = torch.autograd.grad(grad_dot_vec, params, retain_graph=False)
        hvp_flat = torch.cat([hvp_i.flatten() for hvp_i in hvp])
        return hvp_flat.detach().cpu().numpy()

    return calc_hvp


def top_k_eigenvalues(model, loss_fn, x, y, k, device):
    params = list(model.parameters())

    n = sum(p.numel() for p in params)
    hvp = hessian_vector_product_function(model, loss_fn, x, y, device, params)
    H_op = LinearOperator((n, n), matvec=hvp)

    # Compute the top k eigenvalues by magnitude
    eigenvalues, _ = eigsh(H_op, k=k, which='LM')

    return eigenvalues.real

def gradient_hessian_gradient_dot(model, loss_fn, x, y):
    loss = loss_fn(model(x), y)
    params = list(model.parameters())
    
    grad_params = torch.autograd.grad(loss, params, create_graph=True)
    grad_params_flat = torch.cat([g.flatten() for g in grad_params])
    vec = grad_params_flat.detach()

    grad_dot_vec = torch.dot(grad_params_flat, vec)
    hvp = torch.autograd.grad(grad_dot_vec, params, retain_graph=False)
    hvp_flat = torch.cat([hvp_i.flatten() for hvp_i in hvp])
    
    return torch.dot(hvp_flat, vec)

def train_until_convergence(model, x, y, loss_fn, lr, threshold):
    # to be honest, I think this function needs to be replaced with a smarter method
    # Clone the model
    cloned_model = copy.deepcopy(model)

    # Set the optimizer and the learning rate
    optimizer = torch.optim.Adam(cloned_model.parameters(), lr=lr)

    # Train the model until the norm of the gradient drops below the threshold
    epoch = 0
    try:
        for epoch in range(1_000_000):
            # Compute the gradient of the loss
            optimizer.zero_grad()
            loss = loss_fn(cloned_model(x), y)
            loss.backward()

            # Check if the loss is below the threshold
    #         if loss.item() < threshold:
    #             break

            # Update the model parameters
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"{epoch = } {loss.item() = } {threshold = }")
    #     else:
    #         raise ValueError("Never reached the threshold")
    except KeyboardInterrupt:
        print("Interrupting")

    finally:
        optimizer.zero_grad()
        print(f"took {epoch} steps to reach threshold")
        return cloned_model


def trainer(model: nn.Module, optimizer, epochs, time, loader_train,
            loader_test, r, n_c, label_noise, hessian_samples,
            projection_lr, projection_threshold, device, lr_schedule_lambda=lambda epoch: 1,
            project_during=False, project_last=False, name=''):
    # Make data dic, contains training data
    data = {'loss': [], 'test_loss': [], 'ws': [], 'test_acc': [],
            'train_acc': [], 'gradfc1': [], 'gradfc2': [], 'fc1.ws': [],
            'fc2.ws': [], 'grokk_time': [], 'hessian': [], 'valley_hessian': [],
            'hessian_eigs': [], 'valley_hessian_eigs': [], 'grad_norm': []}

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    criterion1 = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_schedule_lambda)
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    stopwatch = 0
    for e in range(epochs):
        data['grokk_time'] = stopwatch
        stopwatch += 1

        for t, (x, y, y_one_hot) in enumerate(loader_train):
            model.train()  # put model to training mode

            model = model.to(device=device)
            scores = model(x)

            loss = criterion1(scores,
                              y_one_hot + label_noise * torch.randn(size=y_one_hot.shape,
                                                                    dtype=dtype,
                                                                    device=device))

            if math.isnan(float(loss.item())):
                raise ValueError("Nan Loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            Test = check_accuracy(loader_test, model)[2]*100
            if Test > 90:
                return data, e
            
            with torch.no_grad():
                Test = check_accuracy(loader_test, model)[2]*100
                Train = check_accuracy(loader_train, model)[2]*100
                data['test_acc'].append(Test)
                data['train_acc'].append(Train)
                data['test_loss'].append(
                    test_loss(model, loader_test).item())
                data['loss'].append(loss.item())

            if e % 1000 == 0 and t == 0:
                with torch.no_grad():
                    # Test = check_accuracy(loader_test, model)[2]*100
                    # Train = check_accuracy(loader_train, model)[2]*100
                    # data['test_acc'].append(Test)
                    # data['train_acc'].append(Train)
                    # data['test_loss'].append(
                    #     test_loss(model, loader_test, n_c).detach().clone().cpu())
                    # data['loss'].append(loss.detach().clone().cpu())

                    grad_norm = 0.0
                    
                    for parameter in model.parameters():
                        grad_norm += torch.sum(torch.square(parameter.grad))
                    
                    data['grad_norm'].append(grad_norm.item()**.5)

                
                hessian_trace_approx = 0.0
                for _ in range(hessian_samples):
                    hessian_trace_approx += hessian_trace(
                        model, criterion1, x, y_one_hot) / hessian_samples

                data['hessian'].append(hessian_trace_approx)
                data['hessian_eigs'].append(top_k_eigenvalues(
                    model, criterion1, x, y_one_hot, 5, device))

                if project_during:
                    if e >= 1000:
                        valley_model = train_until_convergence(
                            model, x, y_one_hot, criterion1, projection_lr, projection_threshold)
                        hessian_trace_approx = 0.0
                        for _ in range(hessian_samples):
                            hessian_trace_approx += hessian_trace(
                                valley_model, criterion1, x, y_one_hot) / hessian_samples

                        data['valley_hessian'].append(hessian_trace_approx)
                        data['valley_hessian_eigs'].append(top_k_eigenvalues(
                            valley_model, criterion1, x, y_one_hot, 5, device))
                    else:
                        data['valley_hessian'].append(float('nan'))

            if e % 1000 == 0 and t == 0:
                with torch.no_grad():
                    print(f'Epoch {(e+1)}' + name + f', Loss = {loss.item()}')
                    print(f'Test acc: {Test}% correctly')
                    print(f'Train acc: {Train}% correctly')
                    print(f'{max(data["hessian_eigs"][-1]) = }')

    if project_last:
        hessian_trace_approx = 0.0
        for _ in range(hessian_samples):
            hessian_trace_approx += hessian_trace(
                model, criterion1, x, y_one_hot) / hessian_samples

        data['hessian'].append(hessian_trace_approx)
        data['hessian_eigs'].append(top_k_eigenvalues(
            model, criterion1, x, y_one_hot, 5, device))

        valley_model = train_until_convergence(
            model, x, y_one_hot, criterion1, projection_lr, projection_threshold)
        hessian_trace_approx = 0.0
        for _ in range(hessian_samples):
            hessian_trace_approx += hessian_trace(
                valley_model, criterion1, x, y_one_hot) / hessian_samples

        data['valley_hessian'].append(hessian_trace_approx)
        data['valley_hessian_eigs'].append(top_k_eigenvalues(
            valley_model, criterion1, x, y_one_hot, 5, device))

    return data, float('inf')


################ MLP model #################
def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    # "flatten" the C * H * W values into a single vector per image
    return x.view(N, -1)


class fcc(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(fcc, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, h_size, bias=False)
        self.fc2 = nn.Linear(h_size, out_size, bias=False)

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0**0.5)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0**0.5)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        # First layer
        x = flatten(x)
        x = math.sqrt(1 / self.in_size) * self.fc1(x)
        x = x**2
        x = math.sqrt(1 / self.h_size) * self.fc2(x)
        return x

    def func(self, x, layer1_w, layer2_w):
        # if I want to evaluate the model at some other point
        x = flatten(x)
        x = math.sqrt(1 / self.in_size) * layer1_w @ x
        x = x**2
        x = math.sqrt(1 / self.h_size) * layer2_w @ x
        return x

class SovlerInitNet(nn.Module):
    def __init__(self, p):
        super(SovlerInitNet, self).__init__()
        hidden_layer_size = p**2 + 2 * p
        layer1init = np.zeros((hidden_layer_size, 2*p), dtype=np.float32)
        layer2init = np.zeros((p, hidden_layer_size), dtype=np.float32)


        for i in range(p):
            for j in range(p):
                layer1init[i + p * j][i] = 1.0
                layer1init[i + p * j][p + j] = 1.0
        
        for i in range(2 * p):
            layer1init[p**2 + i][i] = 1.0
        
        for outp in range(p):
            for i in range(p):
                j = (outp - i) % p
                layer2init[outp][i * p + j] = 1.0
                layer2init[outp][p**2 + i] = -1.0
                layer2init[outp][p**2 + p + j] = -1.0

        self.layer1 = torch.nn.Parameter(torch.tensor(layer1init))
        self.layer2 = torch.nn.Parameter(torch.tensor(layer2init))
    
    def forward(self, x):
        x = flatten(x)
        x = x.T
        x = self.layer1 @ x
        x = x**2
        x = self.layer2 @ x / 2
        return x.T

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('p', type=int, help='The (prime) number to use as the base for the modular addition.')
    parser.add_argument('alpha', type=float, help='The fraction of the data to use for training.')
    parser.add_argument('width', type=int,  help='The width of the hidden layer.')
    parser.add_argument('lr', type=float, help='The learning rate to train with')
    parser.add_argument('beta', type=float, help='The momentum to train with.')
    parser.add_argument('epsilon', type=float, help='The level of algorithmic label noise.')
    parser.add_argument('save_file', type=str, help='The path to save the training data and final model to.')
    parser.add_argument('--epochs', type=int, default=10_000_000, help='The max number of epochs to train for. Defaults to 10M.')
    parser.add_argument('--project', action='store_true', help='Toggle this flag to include periodic projections during training.')
    parser.add_argument('--proj_steps', type=int, default=1_000_000, help='The number of steps to use in the projection step')
    parser.add_argument('--projection_lr', type=float, default=2e2, help='The learning rate to use during the projection step. Defaults to 2e2')
    parser.add_argument('--projection_threshold', type=float, default=np.inf, help='The threshold of loss to stop the projection at. Defaults to infinity.')
    parser.add_argument('--silent', action='store_true', help='Prints minimally and does not dispay plots.')
    parser.add_argument('--random_seed', type=int, default=1, help='The random seed to use to generate the data')
    parser.add_argument('--device', type=str, default='cuda', help='The device to train on. Defaults to "cuda".')

    args = parser.parse_args()

    device = args.device
    print('using device:', args.device)

    loader_test, loader_train = make_data(args.p, args.p, args.alpha, args.device, args.random_seed)

    model = fcc(args.p * 2, args.width, args.p)
    SGD = optim.SGD(model.parameters(), args.lr, momentum=args.beta)

    train_data, grokking_time = trainer(model, SGD, args.epochs, 0, loader_train, loader_test, None, args.p,
                                        args.epsilon, 10, args.projection_lr, args.projection_threshold,
                                        device, project_during=args.project, project_last=args.project)
    
    basepath = pathlib.Path(args.save_file)

    print("Saving Data")
    with basepath.open('wb') as f:
        pickle.dump((train_data, grokking_time), f)
    
    print("Saving Model")
    with (basepath.with_suffix('.model')).open('wb') as f:
        torch.save((model, SGD), f)


if __name__ == '__main__':
    main()