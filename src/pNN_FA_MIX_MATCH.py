import numpy as np
import torch
from FaultAnalysisDropout.PLNC_FA_NAS import *
import random

# ================================================================================================================================================
# ===============================================================  Printed Layer  ================================================================
# ================================================================================================================================================


class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, ACT, INV):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        self.N_fault = args.N_fault

        # define nonlinear circuits
        # self.ACT = ACT
        self.ACT = ACT
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin

        # eta_2 = ACT.eta.mean(0)[2].detach().item()
        # not sure????
        eta_2 = ACT[0].eta_fault[0, 2].detach().item()
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = eta_2 / (1.-eta_2) * \
            (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

        # cofficients for mixture of activation functions
        if (args.type_nonlinear == 'robust' or args.type_nonlinear == 'normal') and args.act != 'none':
            num_act = len(ACT)
            cofficients_act = torch.ones([num_act, n_out])
            self.cofficients_act = torch.nn.Parameter(
                cofficients_act, requires_grad=False)

            num_neg = len(INV)
            cofficients_neg = torch.ones([num_neg, n_in+2])
            self.cofficients_neg = torch.nn.Parameter(
                cofficients_neg, requires_grad=False)
        else:
            num_act = len(ACT)
            cofficients_act = torch.nn.init.uniform_(
                torch.empty(num_act, n_out), a=-0.5, b=0.5)
            self.cofficients_act = torch.nn.Parameter(
                cofficients_act, requires_grad=True)

            num_neg = len(INV)
            cofficients_neg = torch.nn.init.uniform_(
                torch.empty(num_neg, n_in+2), a=-0.5, b=0.5)
            self.cofficients_neg = torch.nn.Parameter(
                cofficients_neg, requires_grad=True)

        self.act_temperature = 1.0  # Temperature parameter for Gumbel-Softmax
        self.neg_temperature = 1.0  # Temperature parameter for Gumbel-Softmax

        self.FaultMask = torch.ones_like(
            self.theta_).repeat(self.N_fault, 1, 1)

        self.FaultMaskACT = [torch.zeros(n_out).repeat(
            self.N_fault, 1, 1) for _ in range(len(self.ACT))]
        self.FaultMaskNEG = [torch.zeros(
            n_in+2).repeat(self.N_fault, 1, 1) for _ in range(len(self.INV))]

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def theta_noisy(self):
        theta_fault = self.theta
        mean = theta_fault.repeat(self.N_fault, self.N, 1, 1)
        nosie = ((torch.rand(mean.shape) * 2.) - 1.) * self.epsilon + 1.
        fault_mask = self.FaultMask.repeat(self.N, 1, 1, 1).permute(1, 0, 2, 3)
        return mean * nosie * fault_mask

    @property
    def W(self):
        # to deal with case that the whole colume of theta is 0
        G = torch.sum(self.theta_noisy.abs(), axis=2, keepdim=True)
        W = self.theta_noisy.abs() / (G + 1e-10)
        return W.to(self.device)

    # @property
    # def CoFF_ACT(self):
    #     self.cofficients_act.data.clamp_(-1, 1)
    #     # to deal with case that the whole colume of theta is 0
    #     cofficients_act_temp = self.cofficients_act.clone()
    #     sum_alpha = torch.sum(cofficients_act_temp, axis=0)
    #     alpha = torch.exp(cofficients_act_temp) / (sum_alpha + 1e-10)
    #     alpha = torch.zeros_like(cofficients_act_temp).scatter_(
    #         0, torch.argmax(alpha, dim=0).unsqueeze(0), 1.)
    #     print('check the values of cofficients: ', alpha)
    #     return alpha.detach() + self.cofficients_act - self.cofficients_act.detach()

    @property
    def CoFF_ACT(self):
        if len(self.ACT) == 1:
            return self.cofficients_act
        # Clamp coefficients to prevent extreme values
        self.cofficients_act.data.clamp_(-1, 1)

        # Apply Gumbel-Softmax trick
        gumbel_noise = - \
            torch.log(-torch.log(torch.rand_like(self.cofficients_act) + 1e-20) + 1e-20)
        gumbel_logits = (self.cofficients_act +
                         gumbel_noise) / self.act_temperature

        # Softmax with temperature to approximate hard selection
        alpha = torch.nn.functional.softmax(gumbel_logits, dim=0)

        # Convert to hard one-hot representation but keep gradients
        alpha_hard = torch.zeros_like(alpha).scatter_(
            0, torch.argmax(alpha, dim=0).unsqueeze(0), 1.0)

        # Use the straight-through estimator: alpha_hard for forward pass, alpha for backward pass
        alpha_straight_through = alpha_hard.detach() + alpha - alpha.detach()
        # print('check the values of coefficients: ', alpha_hard)
        return alpha_straight_through

    @property
    def CoFF_NEG(self):
        if len(self.INV) == 1:
            return self.cofficients_neg
        self.cofficients_neg.data.clamp_(-1, 1)

        gumbel_noise = - \
            torch.log(-torch.log(torch.rand_like(self.cofficients_neg) + 1e-20) + 1e-20)
        gumbel_logits = (self.cofficients_neg +
                         gumbel_noise) / self.neg_temperature

        beta = torch.nn.functional.softmax(gumbel_logits, dim=0)

        beta_hard = torch.zeros_like(beta).scatter_(
            0, torch.argmax(beta, dim=0).unsqueeze(0), 1.0)

        beta_straight_through = beta_hard.detach() + beta - beta.detach()
        # print('neg: check the values of coefficients: ', beta_hard)
        return beta_straight_through

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta_noisy.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        ones_tensor = torch.ones(
            [a.shape[0], a.shape[1], a.shape[2], 1]).to(self.device)
        zeros_tensor = torch.zeros_like(ones_tensor).to(self.device)
        a_extend = torch.cat([a, ones_tensor, zeros_tensor], dim=3)

        # print('shape of neg: ', len(self.FaultMaskNEG), len(self.INV))
        for i, _ in enumerate(self.INV):
            self.INV[i].Mask = self.FaultMaskNEG[i]
        # print("check self.FaultMaskNEG ", self.FaultMaskNEG.shape)
        # print("check MAC ", self.INV.Mask.shape, a_extend.shape)
        beta = self.CoFF_NEG
        a_neg = 0
        for i, inv in enumerate(self.INV):
            a_neg += beta[i, :]*inv(a_extend)
        a_neg[:, :, :, -1] = torch.tensor(0.).to(self.device)

        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)

        # print('check MAC output', z.shape)

        return z

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.mac_power = self.MAC_power(a_previous, z_new)
        for i, _ in enumerate(self.ACT):
            self.ACT[i].Mask = self.FaultMaskACT[i]
        # print("check self.FaultMaskACT ", self.FaultMaskACT.shape)
        alpha = self.CoFF_ACT
        a_new = 0
        for i, act in enumerate(self.ACT):
            a_new += alpha[i, :]*act(z_new)
        return a_new

    @property
    def g_tilde(self):
        # scaled conductances
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MAC_power(self, x, y):
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], x.shape[1], x.shape[2], 1]).to(
                                  self.device),
                              torch.zeros([x.shape[0], x.shape[1], x.shape[2], 1]).to(self.device)], dim=3)

        for i, _ in enumerate(self.INV):
            self.INV[i].Mask = self.FaultMaskNEG[i]

        beta = self.CoFF_NEG
        x_neg = 0
        for i, inv in enumerate(self.INV):
            x_neg += beta[i, :]*inv(x_extend)

        x_neg[:, :, :, -1] = 0.

        F = x_extend.shape[0]
        V = x_extend.shape[1]
        E = x_extend.shape[2]
        M = x_extend.shape[3]
        N = y.shape[3]

        positive = self.theta_noisy.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)
        for f in range(F):
            for v in range(V):
                for m in range(M):
                    for n in range(N):
                        Power += self.g_tilde[m, n] * ((x_extend[f, v, :, m]*positive[f, v, m, n] +
                                                       x_neg[f, v, :, m]*negative[f, v, m, n])-y[f, v, :, n]).pow(2.).sum()
        Power = Power / E / V / F
        return Power

    @property
    def soft_num_theta(self):
        # forward pass: number of theta
        nonzero = self.theta.clone().detach().abs()
        nonzero[nonzero > 0] = 1.
        N_theta = nonzero.sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = torch.sigmoid(self.theta.abs())
        soft_count = soft_count * nonzero
        soft_count = soft_count.sum()
        return N_theta.detach() + soft_count - soft_count.detach()

    @property
    def soft_num_act(self):
        # forward pass: number of act
        nonzero = self.theta.clone().detach().abs()[:-2, :]
        coff_act = self.CoFF_ACT.clone().detach()
        nonzero[nonzero > 0] = 1.
        # print('check dimension of nonzero',
        #       nonzero.shape, self.cofficients_act[0, :])
        N_act = [(coff_act[i, :]*nonzero).max(0)
                 [0].sum() for i in range(len(self.ACT))]

        # backward pass: pvalue of the minimal negative weights
        soft_count = torch.sigmoid(self.theta.abs()[:-2, :])
        soft_count_act = [(soft_count * (self.CoFF_ACT[i, :] * nonzero)
                           ).max(0)[0].sum() for i in range(len(self.ACT))]

        out = [N_act[i].detach() + soft_count_act[i] - soft_count_act[i].detach()
               for i in range(len(self.ACT))]
        # print('check the values of activation functions: ', out)
        return out

    @property
    def act_power(self):
        # need to change
        # wich activation function is used
        # decide based on the value of cofecients acts parameters
        act_power = 0
        for i, _ in enumerate(self.ACT):
            act_power += self.soft_num_act[i].item() * self.ACT[i].power * 1e-6
        return act_power

    @property
    def neg_power(self):
        # need to change
        # wich activation function is used
        # decide based on the value of cofecients acts parameters
        neg_power = 0
        for i, _ in enumerate(self.INV):
            neg_power += self.soft_num_neg[i].item() * self.INV[i].power * 1e-6
        return neg_power

    @property
    def soft_num_neg(self):
        # forward pass: number of negative weights
        positive = self.theta.clone().detach()[:-2, :]
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        coff_neg = self.CoFF_NEG.clone().detach()[:, :-2]

        N_neg = [(coff_neg[i, :] * negative.T).T.max(1)[0].sum()
                 for i in range(len(self.INV))]
        # backward pass: pvalue of the minimal negative weights
        soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
        coff = self.CoFF_NEG[:, :-2]
        soft_count_neg = [(soft_count * (coff[i, :] * negative.T).T
                           ).max(1)[0].sum() for i in range(len(self.INV))]
        out = [N_neg[i].detach() + soft_count_neg[i] - soft_count_neg[i].detach()
               for i in range(len(self.INV))]
        return out

    @property
    def DeviceCount(self):
        # need to change
        # wich activation function is used
        # decide based on the value of cofecients acts parameters
        act_deviceCount = torch.tensor(0.).to(self.device)
        for i, _ in enumerate(self.ACT):
            act_deviceCount += self.soft_num_act[i].item() * \
                self.ACT[i].DeviceCount

        neg_deviceCount = torch.tensor(0.).to(self.device)
        for i, _ in enumerate(self.INV):
            neg_deviceCount += self.soft_num_neg[i].item() * \
                self.INV[i].DeviceCount
        return self.soft_num_theta + act_deviceCount + neg_deviceCount

    def MakeFaultSample(self, e_fault):
        # need to change wich activation function is used
        # if there is need an average of the number of devices in the activation function
        # decide based on the value of cofecients acts parameters

        act_deviceCount = 0
        for i, _ in enumerate(self.ACT):
            act_deviceCount += self.soft_num_act[i].item() * \
                self.ACT[i].DeviceCount

        neg_deviceCount = 0
        for i, _ in enumerate(self.INV):
            neg_deviceCount += self.soft_num_neg[i].item() * \
                self.INV[i].DeviceCount

        TotalDeviceCount = [
            self.soft_num_theta.item(), act_deviceCount, neg_deviceCount]

        indices = list(range(len(TotalDeviceCount)))
        total = sum(TotalDeviceCount)
        probabilities = [num / total for num in TotalDeviceCount]
        fault_sample = random.choices(
            indices, weights=probabilities, k=e_fault)
        # static aftpg
        fault_sample = [1]
        print('which type of faults in this layer are faulty (0:weight, 1:act, 2:neg) :', fault_sample)

        values = [0, 1, 2]
        counts = [fault_sample.count(value) for value in values]

        limitation = [10**10, self.theta_.shape[1], self.theta_.shape[0]-2]
        N_adjust_act = max([0, counts[1] - limitation[1]])
        N_adjust_neg = max([0, counts[2] - limitation[2]])
        counts[0] = counts[0] + N_adjust_act + N_adjust_neg
        counts[1] = counts[1] - N_adjust_act
        counts[2] = counts[2] - N_adjust_neg

        assert sum(
            counts) == e_fault, "Total count should match the number of faults required."

        # flattened_mask = self.FaultMask.flatten()
        flattened_mask = torch.ones(
            self.FaultMask.shape[1] * self.FaultMask.shape[2]).flatten()
        indices_to_modify = torch.randperm(flattened_mask.numel())[:counts[0]]
        # static aftpg
        indices_to_modify = torch.tensor([], dtype=torch.int64)
        print('which weights are faulty: ', indices_to_modify)

        for idx in indices_to_modify:
            flattened_mask[idx] = torch.tensor(
                10.**10. if torch.rand(1).item() > 0.5 else 0., dtype=torch.int64)
        # flattened_mask[indices_to_modify] = torch.where(
        #     torch.rand(counts[0]) > 0.5, 10**10, 0).to(torch.int64)
        FaultMask = flattened_mask.reshape(
            self.FaultMask.shape[1], self.FaultMask.shape[2])

        FaultMaskACT = [torch.zeros(self.theta_.shape[1])
                        for _ in range(len(self.ACT))]
        fault_act = torch.randperm(self.theta_.shape[1])[:counts[1]]
        # static atpg
        fault_act = torch.tensor([1])
        print('which activation functions are faulty: ', fault_act)
        for i in fault_act:
            # As the non-faulty is included in eta-faults so generate eta_fault.shape[0] instead of eta_fault.shape[0]+1
            for j, _ in enumerate(self.ACT):
                FaultMaskACT[j][i] = torch.randint(
                    1, self.ACT[j].eta_fault.shape[0], (1,)).item()

        FaultMaskNEG = [torch.zeros(self.theta_.shape[0])
                        for _ in range(len(self.INV))]
        fault_neg = torch.randperm(self.theta_.shape[0])[:counts[2]]
        # static atpg
        fault_neg = torch.tensor([], dtype=torch.int64)
        print('which negative weights are faulty: ', fault_neg)
        for i in fault_neg:
            # As the non-faulty is included in eta-faults so generate eta_fault.shape[0] instead of eta_fault.shape[0]+1
            for j, _ in enumerate(self.INV):
                FaultMaskNEG[j][i] = torch.randint(
                    1, self.INV[j].eta_fault.shape[0], (1,)).item()

        return FaultMask, FaultMaskACT, FaultMaskNEG

    def MakeFault(self, e_fault):
        if e_fault == 0:
            self.RemoveFault()
        else:
            FaultMask = []
            FaultMaskACT = [[] for _ in range(len(self.ACT))]
            FaultMaskNEG = [[] for _ in range(len(self.INV))]
            for i in range(self.N_fault):
                faultmask, faultmaskact, faultmaskneg = self.MakeFaultSample(
                    e_fault)
                FaultMask.append(faultmask)
                for j, _ in enumerate(self.ACT):
                    FaultMaskACT[j].append(faultmaskact[j])
                for j, _ in enumerate(self.INV):
                    FaultMaskNEG[j].append(faultmaskneg[j])
            self.FaultMask = torch.stack(FaultMask)
            for i, _ in enumerate(self.FaultMaskACT):
                self.FaultMaskACT[i] = torch.stack(FaultMaskACT[i])
            for i, _ in enumerate(self.FaultMaskNEG):
                self.FaultMaskNEG[i] = torch.stack(FaultMaskNEG[i])

    def RemoveFault(self):
        self.FaultMask = torch.ones_like(
            self.theta_).repeat(self.N_fault, 1, 1)
        for i, _ in enumerate(self.FaultMaskACT):
            self.FaultMaskACT[i] = torch.zeros(
                self.theta_.shape[1]).repeat(self.N_fault, 1, 1)
        for i, _ in enumerate(self.FaultMaskNEG):
            self.FaultMaskNEG[i] = torch.zeros(
                self.theta_.shape[0]).repeat(self.N_fault, 1, 1)

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        for i, _ in enumerate(self.INV):
            self.INV[i].N = N
            self.INV[i].epsilon = epsilon


# ================================================================================================================================================
# ==============================================================  Printed Circuit  ===============================================================
# ================================================================================================================================================

class pNN(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()

        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        self.e_fault = args.e_fault
        self.N_fault = args.N_fault
        self.topology = topology

        # define nonlinear circuits
        # self.act = TanhRT(args)
        self.act = []
        self.inv = []
        if (args.type_nonlinear == 'robust' or args.type_nonlinear == 'normal') and args.act == 'none':
            self.act1 = TanhRT(args)
            self.act2 = pReLURT(args)
            self.act3 = SigmoidRT(args)
            self.act4 = ClippedReLU(args)
            self.inv1 = InvRT(args)

            self.act = [self.act1, self.act2, self.act3, self.act4]
            self.inv = [self.inv1]
        elif args.type_nonlinear == 'mix' and args.act == 'none':
            args.type_nonlinear = 'robust'
            self.act1 = TanhRT(args)
            self.act2 = pReLURT(args)
            self.act3 = SigmoidRT(args)
            self.act4 = ClippedReLU(args)
            self.inv1 = InvRT(args)

            args.type_nonlinear = 'normal'
            self.act5 = TanhRT(args)
            self.act6 = pReLURT(args)
            self.act7 = SigmoidRT(args)
            self.act8 = ClippedReLU(args)
            self.inv2 = InvRT(args)

            self.act = [self.act1, self.act2, self.act3, self.act4,
                        self.act5, self.act6, self.act7, self.act8]
            self.inv = [self.inv1, self.inv2]
        else:
            if args.act == 'tanh':
                self.act = [TanhRT(args)]
            elif args.act == 'relu':
                self.act = [pReLURT(args)]
            elif args.act == 'sigmoid':
                self.act = [SigmoidRT(args)]
            elif args.act == 'cr':
                self.act = [ClippedReLU(args)]
            self.inv = [InvRT(args)]

        # area
        self.area_theta = torch.tensor(args.area_theta).to(self.device)
        # self.area_act = torch.tensor(args.area_act).to(self.device)
        # self.area_neg = torch.tensor(args.area_neg).to(self.device)

        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            # print('check topology: ', i, topology[i], topology[i+1])
            self.model.add_module(
                f'{i}-th Layer', pLayer(topology[i], topology[i+1], args, self.act, self.inv))

    def forward(self, x):
        self.RemoveFault()
        self.SampleFault()
        for l in self.model:
            l.act_temperature = max(l.act_temperature * 0.95, 0.1)
            l.neg_temperature = max(l.neg_temperature * 0.95, 0.1)
        x = x.repeat(self.N_fault, self.N, 1, 1)
        out = self.model(x)
        # print('check the output of the model: ', out.shape, x.shape)
        # print("forward pass ----- ", x.shape, out.shape,
        #       self.args.N_fault, self.args.e_fault, self.N_fault, self.e_fault)
        # print(self.N_fault, self.e_fault)
        return out

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def soft_count_neg(self):
        soft_count = [torch.tensor([0.]).to(self.device)
                      for _ in range(len(self.inv))]
        for l in self.model:
            if hasattr(l, 'soft_num_neg'):
                for j, _ in enumerate(self.inv):
                    soft_count[j] += l.soft_num_neg[j].item()
        return soft_count

    @property
    def soft_count_act(self):
        soft_count = [torch.tensor([0.]).to(self.device)
                      for _ in range(len(self.act))]
        for l in self.model:
            if hasattr(l, 'soft_num_act'):
                for j, _ in enumerate(self.act):
                    soft_count[j] += l.soft_num_act[j].item()
        return soft_count

    @property
    def soft_count_theta(self):
        soft_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_theta'):
                soft_count += l.soft_num_theta
        return soft_count

    @property
    def power_neg(self):
        # convert uW to W
        power_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'neg_power'):
                power_count += l.neg_power
        return power_count

    @property
    def power_act(self):
        power_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'act_power'):
                power_count += l.act_power
        return power_count

    @property
    def power_mac(self):
        power_mac = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'mac_power'):
                power_mac += l.mac_power
        return power_mac

    @property
    def Power(self):
        return self.power_neg + self.power_act + self.power_mac

    @property
    def Power_ACT_NEG(self):
        # print('---------- check power ----------',
        #       self.power_neg, self.power_act, self.power_mac)
        return self.power_neg + self.power_act

    @property
    def AVE_POWER_ACT_NEG(self):
        ave_power = 0
        ave_neg_power, ave_act_power = [], []
        num_neg, num_act = 0, 0

        for i, act in enumerate(self.act):
            ave_act_power.append(act.power_fault[0])
            num_act += self.soft_count_act[i].clone().detach().item()
            # Return the middle value
        if len(ave_act_power) % 2 == 0:  # Even number of elements
            ave_act_power.append(0)
        ave_act_power = np.median(ave_act_power)
        ave_power = ave_act_power * num_act * 1e-6

        for i, inv in enumerate(self.inv):
            ave_neg_power.append(inv.power_fault[0])
            num_neg += self.soft_count_neg[i].clone().detach().item()
        if len(ave_neg_power) % 2 == 0:  # Even number of elements
            ave_neg_power.append(0)
        ave_neg_power = np.median(ave_neg_power)
        ave_power += ave_neg_power * num_neg * 1e-6

        # ave_power += self.power_mac.clone().detach().item()

        return ave_power

    @property
    def Area(self):
        act_area = 0
        for i, act in enumerate(self.act):
            act_area += self.soft_count_act[i] * act.Area

        act_inv = 0
        for i, inv in enumerate(self.inv):
            act_inv += self.soft_count_neg[i] * inv.Area

        return act_inv + act_area + self.area_theta * self.soft_count_theta

    @property
    def Area_ACT_NEG(self):
        act_area = 0
        for i, act in enumerate(self.act):
            act_area += self.soft_count_act[i] * act.Area

        act_inv = 0
        for i, inv in enumerate(self.inv):
            act_inv += self.soft_count_neg[i] * inv.Area

        return act_inv + act_area

    @property
    def AVE_AREA_ACT_NEG(self):
        ave_area = 0
        ave_neg_area, ave_act_area = [], []
        num_neg, num_act = 0, 0

        for i, act in enumerate(self.act):
            ave_act_area.append(act.Area)
            num_act += self.soft_count_act[i].clone().detach().item()
        if len(ave_act_area) % 2 == 0:  # Even number of elements
            ave_act_area.append(0)
        ave_act_area = np.median(ave_act_area)
        ave_area = ave_act_area * num_act

        for i, inv in enumerate(self.inv):
            ave_neg_area.append(inv.Area)
            num_neg += self.soft_count_neg[i].clone().detach().item()
        if len(ave_neg_area) % 2 == 0:  # Even number of elements
            ave_neg_area.append(0)
        ave_neg_area = np.median(ave_neg_area)
        ave_area += ave_neg_area * num_neg

        # ave_area += self.area_theta * self.soft_count_theta.clone().detach().item()

        return ave_area

    @property
    def ActCount(self):
        dict_act = {}
        for i, act in enumerate(self.act):
            dict_act[act.name] = self.soft_count_act[i]
        return dict_act

    @property
    def DeviceCount(self):
        count_act = 0
        for i, _ in enumerate(self.act):
            count_act += self.soft_count_act[i].item()

        count_neg = 0
        for i, _ in enumerate(self.inv):
            count_neg += self.soft_count_neg[i].item()
        return count_neg + count_act + self.soft_count_theta

    @property
    def DeviceCount1(self):
        TotalDeviceCount = []
        for l in self.model:
            TotalDeviceCount.append(l.DeviceCount.item())
        return sum(TotalDeviceCount)

    def RemoveFault(self):
        for l in self.model:
            if hasattr(l, 'RemoveFault'):
                l.RemoveFault()

    def SampleFault(self):
        if self.e_fault == 0:
            self.N_fault = 1
            self.RemoveFault()
            return
        else:
            TotalDeviceCount = []
            for l in self.model:
                TotalDeviceCount.append(l.DeviceCount.item())

            indices = list(range(len(TotalDeviceCount)))
            total = sum(TotalDeviceCount)
            probabilities = [num / total for num in TotalDeviceCount]
            fault_sample = random.choices(
                indices, weights=probabilities, k=self.e_fault)
            # static atpg
            fault_sample = [1]
            print("faulty layers: ", fault_sample)

            value_counts = torch.tensor(fault_sample).bincount(minlength=1)
            values = [i for i, count in enumerate(value_counts) if count > 0]
            counts = [count.item() for count in value_counts if count > 0]

            for l, n in zip(values, counts):
                self.model[l].MakeFault(n)

    def GetParam(self):
        weights = [p for name, p in self.named_parameters()
                   if name.endswith('.theta_')]
        nonlinear = [p for name, p in self.named_parameters()
                     if name.endswith('.rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

    def UpdateArgs(self, args):
        for i, _ in enumerate(self.act):
            self.act[i].args = args
        for i, _ in enumerate(self.inv):
            self.inv[i].args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        for i, _ in enumerate(self.act):
            self.act[i].N = N
            self.act[i].epsilon = epsilon
        for i, _ in enumerate(self.inv):
            self.inv[i].N = N
            self.inv[i].epsilon = epsilon
        for layer in self.model:
            if hasattr(layer, 'UpdateVariation'):
                layer.UpdateVariation(N, epsilon)

    def UpdateFault(self, N_fault, e_fault):
        self.e_fault = e_fault
        self.N_fault = N_fault
        for layer in self.model:
            layer.N_fault = N_fault


# ================================================================================================================================================
# =============================================================  pNN Loss function  ==============================================================
# ================================================================================================================================================

class pNNLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L

    def CELoss(self, prediction, label):
        fn = torch.nn.CrossEntropyLoss()
        return fn(prediction, label)

    def forward(self, y, label, power_ratio, power_consumption, area_consumption=0, P_avg=0, A_avg=0):
        F = y.shape[0]
        N = y.shape[1]
        loss = torch.tensor(0.).to(self.args.DEVICE)
        if self.args.metric == 'acc':
            for f in range(F):
                for n in range(N):
                    loss += self.CELoss(y[f, n, :, :], label)
        elif self.args.metric == 'maa':
            for f in range(F):
                for n in range(N):
                    loss += self.standard(y[f, n, :, :], label)

        loss = loss / N / F
        # Print the magnitude of the loss and power consumption for comparison
        # print(f"Accuracy-related loss: {loss.item()}")
        # print(f"Power consumption: {power_consumption.item()}")
        # Add the power consumption term to the loss
        # loss += power_ratio * power_consumption.item()
        if self.args.lagrangian_loss:
            # Calculate the augmented Lagrangian penalty terms
            power_penalty = self.args.lambda_P * \
                ((power_consumption - P_avg)) + (self.args.rho / 2) * \
                ((power_consumption - P_avg))**2
            # area_penalty = self.args.lambda_A * \
            #     ((area_consumption - A_avg)) + (self.args.rho / 2) * \
            #     ((area_consumption - A_avg))**2
            # Total loss
            # print('check the loss value and terms: ', loss.item(),
            #       power_consumption - P_avg, (power_consumption - P_avg)**2)
            loss = loss + power_penalty
            # print(loss)
        return loss
