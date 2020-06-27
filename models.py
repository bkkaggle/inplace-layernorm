import torch
import torch.nn as nn
import torch.nn.functional as F


# from http://d2l.ai/chapter_convolutional-modern/batch-norm.html#implementation-from-scratch


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use torch.is_grad_enabled() to determine whether the current mode is
    # training mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is the prediction mode, directly use the mean and variance
        # obtained from the incoming moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # When using a two-dimensional convolutional layer, calculate the
        # mean and variance on the channel dimension (axis=1). Here we
        # need to maintain the shape of X, so that the broadcast operation
        # can be carried out later
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance of the moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims=4, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        shape = (1, num_features, 1, 1)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # The scale parameter and the shift parameter involved in gradient
        # finding and iteration are initialized to 0 and 1 respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # All the variables not involved in gradient finding and iteration are
        # initialized to 0 on the CPU
        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.zeros(shape, device=device)

    def forward(self, X):
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# from https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
class BatchNormFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta):

        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        out = gammax + beta

        ctx.save_for_backward(xhat, gamma, xmu, ivar, sqrtvar, var)

        return out

    @staticmethod
    def backward(ctx, dout):
        xhat, gamma, xmu, ivar, sqrtvar, var = ctx.saved_tensors

        dx = dgamma = dbeta = None

        dbeta = dout.sum(dim=(0, 2, 3), keepdim=True)

        dgammax = dout

        dgamma = torch.sum(dgammax * xhat, dim=(0, 2, 3), keepdim=True)
        dxhat = dgammax * gamma

        divar = torch.sum(dxhat * xmu, dim=(0, 2, 3), keepdim=True)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1.0 / (sqrtvar ** 2) * divar

        dvar = 0.5 * 1.0 / torch.sqrt(var + 1e-5) * dsqrtvar

        dsq = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dvar

        dxmu2 = 2.0 * xmu * dsq

        dx1 = dxmu1 + dxmu2
        dmu = -1.0 * torch.sum(dxmu1 + dxmu2, dim=(0, 2, 3), keepdim=True)

        dx2 = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dmu

        dx = dx1 + dx2

        return dx, dgamma, dbeta


class BatchNormAutograd(nn.Module):
    def __init__(self, num_features):
        super(BatchNormAutograd, self).__init__()

        shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.zeros(shape, device=device)

    def forward(self, x):
        if not torch.is_grad_enabled():
            out = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
            out = self.gamma * out + self.beta

        else:

            out = BatchNormFN.apply(x, self.gamma, self.beta)

            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

                self.moving_mean = 0.9 * self.moving_mean + (1 - 0.9) * mean
                self.moving_var = 0.9 * self.moving_var + (1 - 0.9) * var

        return out


# from https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = grad_output * (input > 0)

        return grad_input


class ReLUAutograd(nn.Module):
    def __init__(self):
        super(ReLUAutograd, self).__init__()

    def forward(self, x):
        x = MyReLU.apply(x)
        return x


class ActivatedBatchNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta):

        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        bn_out = gammax + beta

        out = bn_out.clamp(min=0)

        ctx.save_for_backward(xhat, gamma, xmu, ivar, sqrtvar, var, bn_out)

        return out

    @staticmethod
    def backward(ctx, dout):
        xhat, gamma, xmu, ivar, sqrtvar, var, bn_out = ctx.saved_tensors

        dx = dgamma = dbeta = None

        dout = dout * (bn_out > 0)

        dbeta = dout.sum(dim=(0, 2, 3), keepdim=True)

        dgammax = dout

        dgamma = torch.sum(dgammax * xhat, dim=(0, 2, 3), keepdim=True)
        dxhat = dgammax * gamma

        divar = torch.sum(dxhat * xmu, dim=(0, 2, 3), keepdim=True)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1.0 / (sqrtvar ** 2) * divar

        dvar = 0.5 * 1.0 / torch.sqrt(var + 1e-5) * dsqrtvar

        dsq = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dvar

        dxmu2 = 2.0 * xmu * dsq

        dx1 = dxmu1 + dxmu2
        dmu = -1.0 * torch.sum(dxmu1 + dxmu2, dim=(0, 2, 3), keepdim=True)

        dx2 = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dmu

        dx = dx1 + dx2

        return dx, dgamma, dbeta


class ActivatedBatchNormAutograd(nn.Module):
    def __init__(self, num_features):
        super(ActivatedBatchNormAutograd, self).__init__()

        shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.zeros(shape, device=device)

    def forward(self, x):
        if not torch.is_grad_enabled():
            out = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
            out = self.gamma * out + self.beta
            out = out.clamp(min=0)
        else:

            out = ActivatedBatchNorm.apply(x, self.gamma, self.beta)

            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

                self.moving_mean = 0.9 * self.moving_mean + (1 - 0.9) * mean
                self.moving_var = 0.9 * self.moving_var + (1 - 0.9) * var

        return out


class CheckpointABNFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta):

        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        bn_out = gammax + beta

        out = bn_out.clamp(min=0)

        # ctx.save_for_backward(xhat, gamma, xmu, ivar, sqrtvar, var, bn_out)
        ctx.save_for_backward(x, gamma, beta)

        return out

    @staticmethod
    def backward(ctx, dout):
        # xhat, gamma, xmu, ivar, sqrtvar, var, bn_out = ctx.saved_tensors
        x, gamma, beta = ctx.saved_tensors

        # recompute
        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        bn_out = gammax + beta

        # backwards pass

        dx = dgamma = dbeta = None

        dout = dout * (bn_out > 0)

        dbeta = dout.sum(dim=(0, 2, 3), keepdim=True)

        dgammax = dout

        dgamma = torch.sum(dgammax * xhat, dim=(0, 2, 3), keepdim=True)
        dxhat = dgammax * gamma

        divar = torch.sum(dxhat * xmu, dim=(0, 2, 3), keepdim=True)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1.0 / (sqrtvar ** 2) * divar

        dvar = 0.5 * 1.0 / torch.sqrt(var + 1e-5) * dsqrtvar

        dsq = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dvar

        dxmu2 = 2.0 * xmu * dsq

        dx1 = dxmu1 + dxmu2
        dmu = -1.0 * torch.sum(dxmu1 + dxmu2, dim=(0, 2, 3), keepdim=True)

        dx2 = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dmu

        dx = dx1 + dx2

        return dx, dgamma, dbeta


class CheckpointABN(nn.Module):
    def __init__(self, num_features):
        super(CheckpointABN, self).__init__()

        shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.zeros(shape, device=device)

    def forward(self, x):
        if not torch.is_grad_enabled():
            out = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
            out = self.gamma * out + self.beta
            out = out.clamp(min=0)
        else:

            out = CheckpointABNFN.apply(x, self.gamma, self.beta)

            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

                self.moving_mean = 0.9 * self.moving_mean + (1 - 0.9) * mean
                self.moving_var = 0.9 * self.moving_var + (1 - 0.9) * var

        return out


class CheckpointBNFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta):

        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        bn_out = gammax + beta

        # ctx.save_for_backward(xhat, gamma, xmu, ivar, sqrtvar, var, bn_out)
        ctx.save_for_backward(x, gamma, beta)

        return bn_out

    @staticmethod
    def backward(ctx, dout):
        # xhat, gamma, xmu, ivar, sqrtvar, var, bn_out = ctx.saved_tensors
        x, gamma, beta = ctx.saved_tensors

        # recompute
        mu = x.mean(dim=(0, 2, 3), keepdim=True)

        xmu = x - mu
        sq = xmu ** 2

        var = sq.mean(dim=(0, 2, 3), keepdim=True)

        sqrtvar = torch.sqrt(var + 1e-5)

        ivar = 1.0 / sqrtvar

        xhat = xmu * ivar

        gammax = gamma * xhat

        bn_out = gammax + beta

        # backwards pass

        dx = dgamma = dbeta = None

        dbeta = dout.sum(dim=(0, 2, 3), keepdim=True)

        dgammax = dout

        dgamma = torch.sum(dgammax * xhat, dim=(0, 2, 3), keepdim=True)
        dxhat = dgammax * gamma

        divar = torch.sum(dxhat * xmu, dim=(0, 2, 3), keepdim=True)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1.0 / (sqrtvar ** 2) * divar

        dvar = 0.5 * 1.0 / torch.sqrt(var + 1e-5) * dsqrtvar

        dsq = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dvar

        dxmu2 = 2.0 * xmu * dsq

        dx1 = dxmu1 + dxmu2
        dmu = -1.0 * torch.sum(dxmu1 + dxmu2, dim=(0, 2, 3), keepdim=True)

        dx2 = 1.0 / (dout.shape[0] * dout.shape[2] *
                     dout.shape[3]) * torch.ones_like(dout) * dmu

        dx = dx1 + dx2

        return dx, dgamma, dbeta


class CheckpointBN(nn.Module):
    def __init__(self, num_features):
        super(CheckpointBN, self).__init__()

        shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.moving_mean = torch.zeros(shape, device=device)
        self.moving_var = torch.zeros(shape, device=device)

    def forward(self, x):
        if not torch.is_grad_enabled():
            out = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
            out = self.gamma * out + self.beta
        else:

            out = CheckpointBNFN.apply(x, self.gamma, self.beta)

            with torch.no_grad():
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

                self.moving_mean = 0.9 * self.moving_mean + (1 - 0.9) * mean
                self.moving_var = 0.9 * self.moving_var + (1 - 0.9) * var

        return out


class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, abn_type):
        super().__init__()

        self.abn_type = abn_type

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3)

        if abn_type == 'pt':
            self.abn = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        elif abn_type == 'custom':
            self.abn = nn.Sequential(
                BatchNorm(out_ch),
                nn.ReLU()
            )

        elif abn_type == 'autograd':
            self.abn = nn.Sequential(
                BatchNormAutograd(out_ch),
                nn.ReLU()
            )

        elif abn_type == 'autogradRelu':
            self.abn = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                ReLUAutograd()
            )

        elif abn_type == 'abn':
            self.abn = ActivatedBatchNormAutograd(out_ch)

        elif abn_type == 'checkpoint':
            self.abn = CheckpointABN(out_ch)

        elif abn_type == 'checkpointBN':
            self.abn = CheckpointBN(out_ch)

        self.pool = nn.MaxPool2d((2, 2), 2)

    def forward(self, x):
        x = self.conv(x)

        x = self.abn(x)

        x = self.pool(x)

        return x


class Net(torch.nn.Module):
    def __init__(self, abn_type):
        super().__init__()

        self.block1 = Block(3, 16, abn_type)
        self.block2 = Block(16, 32, abn_type)
        self.block3 = Block(32, 64, abn_type)

        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(x.shape[0], -1)

        x = self.linear(x)

        return x
