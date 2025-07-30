import numpy as np
import torch

# ================================================================================================================================================
# =====================================================  Learnable Negative Weight Circuit  ======================================================
# ================================================================================================================================================


class InvRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        args.N_fault = args.N_fault

        # R1, R2, R3, W1, L1, W2, L2, W3, L3
        self.rt_ = torch.nn.Parameter(torch.tensor([args.NEG_R1n, args.NEG_R2n, args.NEG_R3n, args.NEG_W1n, args.NEG_L1n,
                                      args.NEG_W2n, args.NEG_L2n, args.NEG_W3n, args.NEG_L3n]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/neg_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/neg_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

        self.eta_fault = torch.tensor([[8.3907e-01, -1.0000e+00,  0.0000e+00,  2.0978e-17],
                                       [-6.0647e-01, -1.0000e+00,
                                           0.0000e+00,  1.0867e-08],
                                       [-9.9992e-01, -1.0000e+00,
                                           0.0000e+00, -4.5596e-18],
                                       [8.3907e-01, -1.0000e+00,
                                           0.0000e+00, -1.8533e-17],
                                       [3.1485e+01,  2.8551e-03, -
                                           9.9980e-02,  6.3016e+00],
                                       [-1.0000e+00, -1.0000e+00,
                                           0.0000e+00,  4.4070e-16],
                                       [1.8307e+02, -1.8394e+02, -
                                           8.3575e+01,  4.2966e-02],
                                       [-6.0647e-01, -1.0000e+00,
                                           0.0000e+00,  1.0867e-08],
                                       [1.2159e-01, -7.3578e-01, -
                                           7.9441e-02,  3.1090e+00],
                                       [8.3907e-01, -1.0000e+00,
                                           0.0000e+00,  2.0978e-17],
                                       [-9.9992e-01, -1.0000e+00,
                                           0.0000e+00, -4.5596e-18],
                                       [7.6517e-01, -8.0291e-03,
                                           6.3714e-01,  1.2184e+00],
                                       [8.3907e-01, -1.0000e+00,
                                           0.0000e+00,  2.0978e-17],
                                       [8.3907e-01, -1.0000e+00,
                                           0.0000e+00, -1.8533e-17],
                                       [-1.0000e+00, -1.0000e+00,
                                           0.0000e+00,  4.4070e-16],
                                       [1.0000e+00, -1.0000e+00,
                                           0.0000e+00,  4.4982e-16],
                                       [-4.5913e-02, -7.2633e-01,
                                           7.3493e-02,  1.0507e+01],
                                       [-9.9992e-01, -1.0000e+00,  0.0000e+00, -4.5596e-18]]).to(self.DEVICE)

        self.Mask = None

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([12]).to(self.DEVICE)
        RTn[:9] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:, 0], self.RT_noisy[:, 1], self.RT_noisy[:, 2], self.RT_noisy[:, 3],
                                 self.RT_noisy[:, 4], self.RT_noisy[:,
                                                                    5], self.RT_noisy[:, 6], self.RT_noisy[:, 7],
                                 self.RT_noisy[:, 8], self.RT_noisy[:, 3]/self.RT_noisy[:,
                                                                                        4], self.RT_noisy[:, 5]/self.RT_noisy[:, 6],
                                 self.RT_noisy[:, 7]/self.RT_noisy[:, 8]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def output_variation(self, eta, z):
        # print("output_variation: check output shape ------------------")
        a = torch.zeros_like(z)
        # print("a shape: ", a.shape)
        # print("z shape: ", z.shape)
        # iterate over variations
        for i in range(self.N):
            a[i, :, :] = -(eta[i, 0] + eta[i, 1] *
                           torch.tanh((z[i, :, :] - eta[i, 2]) * eta[i, 3]))
        return a

    def output_faults(self, z, mask):
        slices = []
        # print("output_faults: check output shape ------------------")
        # print("z shape: ", z.shape)
        # print("mask shape: ", mask.shape)
        # for i-th output we have i-th mask
        # so maybe we can apply matrix multiplication instead of iterating over mask
        for i, m in enumerate(mask):
            if not m:
                slices.append(self.output_variation(self.eta, z)[:, :, i])
            else:
                eta_temp = self.eta_fault[int(m-1), :].repeat(self.N, 1)
                slices.append(self.output_variation(eta_temp, z)[:, :, i])

        output = torch.stack(slices, dim=2)
        return output

    def forward(self, z):
        result = []
        # Iterate on every fault
        # print("forward pLNC: check output shape ------------------")
        # print("z shape: ", z.shape)
        # print("Mask shape: ", self.Mask.shape)
        print('check the mask sahp', self.Mask.shape)
        for i in range(self.Mask.shape[0]):
            result.append(self.output_faults(
                z[i, :, :, :], self.Mask[i].flatten()))
        return torch.stack(result)

    def UpdateArgs(self, args):
        self.args = args


# ================================================================================================================================================
# ========================================================  Learnable Activation Circuit  ========================================================
# ================================================================================================================================================

class TanhRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R1n, R2n, W1n, L1n, W2n, L2n
        self.rt_ = torch.nn.Parameter(
            torch.tensor([args.ACT_R1n, args.ACT_R2n, args.ACT_W1n, args.ACT_L1n, args.ACT_W2n, args.ACT_L2n]), requires_grad=True)

        # model
        package = torch.load('./utils/act_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for n, p in self.eta_estimator.named_parameters():
            p.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/act_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        self.power_estimator.train(False)
        for n, p in self.power_estimator.named_parameters():
            p.requires_grad = False
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

        self.eta_fault = torch.tensor([[9.9997e-01, -1.0000e+00,  0.0000e+00,  9.1182e-18],
                                       [-5.5856e-01, -1.0000e+00,
                                           0.0000e+00, -3.7472e-10],
                                       [-1.6919e+01,  1.4562e+01,
                                           8.6479e-02,  8.2578e+01],
                                       [1.0000e+00, -1.0000e+00,
                                           0.0000e+00,  4.4982e-16],
                                       [1.6839e+00, -2.2404e+00, -
                                           2.5173e+00,  1.1147e+00],
                                       [-5.5856e-01, -1.0000e+00,
                                           0.0000e+00, -3.7472e-10],
                                       [9.9997e-01, -1.0000e+00,
                                           0.0000e+00,  9.1182e-18],
                                       [2.3198e-01, -7.8334e-01, -
                                           4.1194e-01,  3.4699e+00],
                                       [1.0000e+00, -1.0000e+00,
                                           0.0000e+00,  4.4982e-16],
                                       [9.9997e-01, -1.0000e+00,
                                           0.0000e+00,  9.1182e-18],
                                       [-2.4548e-01, -2.3790e-02,
                                           8.1499e-01,  1.5337e+00],
                                       [-1.0000e+00, -1.0000e+00,  0.0000e+00,  4.4070e-16]]).to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'TanhRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([9]).to(self.DEVICE)
        RTn[:6] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:, 0], self.RT_noisy[:, 1], self.RT_noisy[:, 2], self.RT_noisy[:, 3],
                                 self.RT_noisy[:, 4], self.RT_noisy[:,
                                                                    5], self.RT_noisy[:, 1]/self.RT_noisy[:, 0],
                                 self.RT_noisy[:, 3]/self.RT_noisy[:, 2], self.RT_noisy[:, 5]/self.RT_noisy[:, 4]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        power_n = self.power_estimator(self.RTn_extend)
        print("power_n shape: ", power_n.shape, self.RTn_extend.shape)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def output_variation(self, eta, z):
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = eta[i, 0] + eta[i, 1] * \
                torch.tanh((z[i, :, :] - eta[i, 2]) * eta[i, 3])
        return a

    def output_faults(self, z, mask):
        slices = []
        for i, m in enumerate(mask):
            if not m:
                slices.append(self.output_variation(self.eta, z)[:, :, i])
            else:
                eta_temp = self.eta_fault[int(m-1), :].repeat(self.N, 1)
                slices.append(self.output_variation(eta_temp, z)[:, :, i])

        output = torch.stack(slices, dim=2)
        return output

    def forward(self, z):
        result = []
        for i in range(self.Mask.shape[0]):
            result.append(self.output_faults(
                z[i, :, :, :], self.Mask[i].flatten()))
        return torch.stack(result)

    def UpdateArgs(self, args):
        self.args = args


# ================================================================================================================================================
# ======================================================  Learnable Hard Sigmoid Activation  =====================================================
# ================================================================================================================================================


class HardSigmoidRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R, W, L
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.HS_Rn, args.HS_Wn, args.HS_Ln]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/hard_sigmoid_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/hard_sigmoid_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def name(self):
        return 'HardSigmoidRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([4]).to(self.DEVICE)
        RTn[:3] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:, 0], self.RT_noisy[:, 1],
                                self.RT_noisy[:, 2], self.RT_noisy[:, 1]/self.RT_noisy[:, 2]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            linear_segment = self.eta[i, 0] + (self.eta[i, 1] - self.eta[i, 0]) / (
                self.eta[i, 3] - self.eta[i, 2]) * (z[i, :, :] - self.eta[i, 2])
            a[i, :, :] = torch.where(z[i, :, :] < self.eta[i, 2], self.eta[i, 0], torch.where(
                z[i, :, :] <= self.eta[i, 3], linear_segment, self.eta[i, 1]))
        return a

    def UpdateArgs(self, args):
        self.args = args


# ================================================================================================================================================
# =======================================================  Learnable Soft pReLU Activation  ======================================================
# ================================================================================================================================================


class pReLURT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # RH, RL, RD, RB, W, L
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.ReLU_RHn, args.ReLU_RLn, args.ReLU_RDn, args.ReLU_RBn, args.ReLU_Wn, args.ReLU_Ln]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/ReLU_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/ReLU_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def name(self):
        return 'pReLURT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([8]).to(self.DEVICE)
        RTn[:6] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:, 0], self.RT_noisy[:, 1], self.RT_noisy[:, 2], self.RT_noisy[:, 3],
                                 self.RT_noisy[:, 4], self.RT_noisy[:, 5], self.RT_noisy[:, 0]/self.RT_noisy[:, 1], self.RT_noisy[:, 4]/self.RT_noisy[:, 5]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        def softplus(x, beta):
            return (1.0 / beta) * torch.log(1 + torch.exp(beta * x))
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = self.eta[i, 0] * (z[i, :, :] - self.eta[i, 2]) + self.eta[i, 1] * softplus(
                z[i, :, :] - self.eta[i, 2], self.eta[i, 4]) + self.eta[i, 3]
        return a

    def UpdateArgs(self, args):
        self.args = args

# ================================================================================================================================================
# ========================================================  Learnable Sigmoid Activation  ========================================================
# ================================================================================================================================================


class SigmoidRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        # R1, R2, W1, L1, W2, L2
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.S_R1n, args.S_R2n, args.S_W1n, args.S_L1n, args.S_W2n, args.S_L2n]).to(args.DEVICE), requires_grad=True)
        # model
        package = torch.load('./utils/sigmoid_param.package')
        self.eta_estimator = package['eta_estimator'].to(self.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.DEVICE)
        self.X_min = package['X_min'].to(self.DEVICE)
        self.Y_max = package['Y_max'].to(self.DEVICE)
        self.Y_min = package['Y_min'].to(self.DEVICE)
        # load power model
        package = torch.load('./utils/sigmoid_power.package')
        self.power_estimator = package['power_estimator'].to(self.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.DEVICE)
        self.pow_X_min = package['X_min'].to(self.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.DEVICE)

    @property
    def name(self):
        return 'SigmoidRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        RTn = torch.zeros([8]).to(self.DEVICE)
        RTn[:6] = rt_temp
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT

    @property
    def RT_noisy(self):
        RT_mean = self.RT.repeat(self.N, 1)
        noise = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = RT_mean * noise
        return RT_variation

    @property
    def RTn_extend(self):
        RT_extend = torch.stack([self.RT_noisy[:, 0], self.RT_noisy[:, 1], self.RT_noisy[:, 2], self.RT_noisy[:, 3],
                                 self.RT_noisy[:, 4], self.RT_noisy[:, 5], self.RT_noisy[:, 2]/self.RT_noisy[:, 3], self.RT_noisy[:, 4]/self.RT_noisy[:, 5]], dim=1)
        return (RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.mean()

    def forward(self, z):
        eta_noisy = self.eta
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = self.eta[i, 0] + self.eta[i, 1] * \
                torch.sigmoid((z[i, :, :] - self.eta[i, 2]) * self.eta[i, 3])
        return a

    def UpdateArgs(self, args):
        self.args = args
