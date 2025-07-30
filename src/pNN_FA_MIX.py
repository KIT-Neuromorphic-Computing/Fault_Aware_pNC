import numpy as np
import torch
from FaultAnalysisDropout.PLNC_FA_NAS import *
import random

# ================================================================================================================================================
# ===============================================================  Printed Layer  ================================================================
# ================================================================================================================================================


class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, ACT1, ACT2, ACT3, ACT4, INV):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        self.N_fault = args.N_fault

        # define nonlinear circuits
        # self.ACT = ACT
        self.ACT1 = ACT1
        self.ACT2 = ACT2
        self.ACT3 = ACT3
        self.ACT4 = ACT4
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin

        # eta_2 = ACT.eta.mean(0)[2].detach().item()
        # not sure????
        eta_2 = ACT1.eta_fault[0, 2].detach().item()
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = eta_2 / (1.-eta_2) * \
            (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

        # cofficients for mixture of activation functions
        cofficients_act = torch.rand([4, n_out])
        self.cofficients_act = torch.nn.Parameter(
            cofficients_act, requires_grad=True)
        self.temperature = 1.0  # Temperature parameter for Gumbel-Softmax

        self.FaultMask = torch.ones_like(
            self.theta_).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT1 = torch.zeros(n_out).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT2 = torch.zeros(n_out).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT3 = torch.zeros(n_out).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT4 = torch.zeros(n_out).repeat(self.N_fault, 1, 1)
        self.FaultMaskNEG = torch.zeros(n_in+2).repeat(self.N_fault, 1, 1)

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
        # Clamp coefficients to prevent extreme values
        self.cofficients_act.data.clamp_(-1, 1)

        # Apply Gumbel-Softmax trick
        gumbel_noise = - \
            torch.log(-torch.log(torch.rand_like(self.cofficients_act) + 1e-20) + 1e-20)
        gumbel_logits = (self.cofficients_act +
                         gumbel_noise) / self.temperature

        # Softmax with temperature to approximate hard selection
        alpha = torch.nn.functional.softmax(gumbel_logits, dim=0)

        # Convert to hard one-hot representation but keep gradients
        alpha_hard = torch.zeros_like(alpha).scatter_(
            0, torch.argmax(alpha, dim=0).unsqueeze(0), 1.0)

        # Use the straight-through estimator: alpha_hard for forward pass, alpha for backward pass
        alpha_straight_through = alpha_hard.detach() + alpha - alpha.detach()
        # print('check the values of coefficients: ', alpha_straight_through)
        return alpha_straight_through

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

        self.INV.Mask = self.FaultMaskNEG
        # print("check self.FaultMaskNEG ", self.FaultMaskNEG.shape)
        # print("check MAC ", self.INV.Mask.shape, a_extend.shape)
        a_neg = self.INV(a_extend)
        a_neg[:, :, :, -1] = torch.tensor(0.).to(self.device)

        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)

        # print('check MAC output', z.shape)

        return z

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.mac_power = self.MAC_power(a_previous, z_new)
        self.ACT1.Mask = self.FaultMaskACT1
        self.ACT2.Mask = self.FaultMaskACT2
        self.ACT3.Mask = self.FaultMaskACT3
        self.ACT4.Mask = self.FaultMaskACT4
        # print("check self.FaultMaskACT ", self.FaultMaskACT.shape)
        alpha = self.CoFF_ACT
        a_new = alpha[0]*self.ACT1(z_new) + alpha[1]*self.ACT2(
            z_new) + alpha[2, :]*self.ACT3(z_new) + alpha[3]*self.ACT4(z_new)
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

        self.INV.Mask = self.FaultMaskNEG
        x_neg = self.INV(x_extend)
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
        nonzero[nonzero > 0] = 1.
        # print('check dimension of nonzero',
        #       nonzero.shape, self.cofficients_act[0, :])
        N_act1 = (
            nonzero * self.CoFF_ACT[0, :].clone().detach()).max(0)[0].sum()
        N_act2 = (
            nonzero * self.CoFF_ACT[1, :].clone().detach()).max(0)[0].sum()
        N_act3 = (
            nonzero * self.CoFF_ACT[2, :].clone().detach()).max(0)[0].sum()
        N_act4 = (
            nonzero * self.CoFF_ACT[3, :].clone().detach()).max(0)[0].sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = torch.sigmoid(self.theta.abs()[:-2, :])
        soft_count1 = soft_count * nonzero * self.CoFF_ACT[0, :]
        soft_count2 = soft_count * nonzero * self.CoFF_ACT[1, :]
        soft_count3 = soft_count * nonzero * self.CoFF_ACT[2, :]
        soft_count4 = soft_count * nonzero * self.CoFF_ACT[3, :]

        soft_count1 = soft_count1.max(0)[0].sum()
        soft_count2 = soft_count2.max(0)[0].sum()
        soft_count3 = soft_count3.max(0)[0].sum()
        soft_count4 = soft_count4.max(0)[0].sum()

        return N_act1.detach() + soft_count1 - soft_count1.detach(), N_act2.detach() + soft_count2 - soft_count2.detach(), N_act3.detach() + soft_count3 - soft_count3.detach(), N_act4.detach() + soft_count4 - soft_count4.detach()
        # return N_act.detach() + soft_count - soft_count.detach()

    @property
    def act_power(self):
        # need to change
        # wich activation function is used
        # decide based on the value of cofecients acts parameters
        act_power = self.soft_num_act[0] * self.ACT1.Power * 1e-6 + \
            self.soft_num_act[1] * self.ACT2.Power * 1e-6 + \
            self.soft_num_act[2] * self.ACT3.Power * 1e-6 + \
            self.soft_num_act[3] * self.ACT4.Power * 1e-6
        return act_power

    @property
    def soft_num_neg(self):
        # forward pass: number of negative weights
        positive = self.theta.clone().detach()[:-2, :]
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        N_neg = negative.max(1)[0].sum()
        # backward pass: pvalue of the minimal negative weights
        soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
        soft_count = soft_count * negative
        soft_count = soft_count.max(1)[0].sum()
        return N_neg.detach() + soft_count - soft_count.detach()

    @property
    def DeviceCount(self):
        # need to change
        # wich activation function is used
        # decide based on the value of cofecients acts parameters
        act_deviceCount = self.soft_num_act[0] * self.ACT1.DeviceCount + \
            self.soft_num_act[1] * self.ACT2.DeviceCount + \
            self.soft_num_act[2] * self.ACT3.DeviceCount + \
            self.soft_num_act[3] * self.ACT4.DeviceCount
        return self.soft_num_theta + act_deviceCount + self.soft_num_neg * self.INV.DeviceCount

    def MakeFaultSample(self, e_fault):
        # need to change wich activation function is used
        # if there is need an average of the number of devices in the activation function
        # decide based on the value of cofecients acts parameters
        act_deviceCount = self.soft_num_act[0] * self.ACT1.DeviceCount + \
            self.soft_num_act[1] * self.ACT2.DeviceCount + \
            self.soft_num_act[2] * self.ACT3.DeviceCount + \
            self.soft_num_act[3] * self.ACT4.DeviceCount
        TotalDeviceCount = [self.soft_num_theta.item(
        ), act_deviceCount, self.soft_num_neg.item() * self.INV.DeviceCount]
        indices = list(range(len(TotalDeviceCount)))
        total = sum(TotalDeviceCount)
        probabilities = [num / total for num in TotalDeviceCount]
        fault_sample = random.choices(
            indices, weights=probabilities, k=e_fault)

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
        for idx in indices_to_modify:
            flattened_mask[idx] = torch.tensor(
                10.**10. if torch.rand(1).item() > 0.5 else 0., dtype=torch.int64)
        # flattened_mask[indices_to_modify] = torch.where(
        #     torch.rand(counts[0]) > 0.5, 10**10, 0).to(torch.int64)
        FaultMask = flattened_mask.reshape(
            self.FaultMask.shape[1], self.FaultMask.shape[2])

        FaultMaskACT1 = torch.zeros(self.theta_.shape[1])
        FaultMaskACT2 = torch.zeros(self.theta_.shape[1])
        FaultMaskACT3 = torch.zeros(self.theta_.shape[1])
        FaultMaskACT4 = torch.zeros(self.theta_.shape[1])
        fault_act = torch.randperm(self.theta_.shape[1])[:counts[1]]
        for i in fault_act:
            # As the non-faulty is included in eta-faults so generate eta_fault.shape[0] instead of eta_fault.shape[0]+1
            FaultMaskACT1[i] = torch.randint(
                1, self.ACT1.eta_fault.shape[0], (1,)).item()
            FaultMaskACT2[i] = torch.randint(
                1, self.ACT2.eta_fault.shape[0], (1,)).item()
            FaultMaskACT3[i] = torch.randint(
                1, self.ACT3.eta_fault.shape[0], (1,)).item()
            FaultMaskACT4[i] = torch.randint(
                1, self.ACT4.eta_fault.shape[0], (1,)).item()

        FaultMaskNEG = torch.zeros(self.theta_.shape[0])
        fault_neg = torch.randperm(self.theta_.shape[0])[:counts[2]]
        for i in fault_neg:
            # As the non-faulty is included in eta-faults so generate eta_fault.shape[0] instead of eta_fault.shape[0]+1
            FaultMaskNEG[i] = torch.randint(
                1, self.INV.eta_fault.shape[0], (1,)).item()

        return FaultMask, FaultMaskACT1, FaultMaskACT2, FaultMaskACT3, FaultMaskACT4, FaultMaskNEG

    def MakeFault(self, e_fault):
        if e_fault == 0:
            self.RemoveFault()
        else:
            FaultMask, FaultMaskACT1, FaultMaskACT2, FaultMaskACT3, FaultMaskACT4, FaultMaskNEG = [
            ], [], [], [], [], []
            for i in range(self.N_fault):
                faultmask, faultmaskact1, faultmaskact2, faultmaskact3, faultmaskact4, faultmaskneg = self.MakeFaultSample(
                    e_fault)
                FaultMask.append(faultmask)
                FaultMaskACT1.append(faultmaskact1)
                FaultMaskACT2.append(faultmaskact2)
                FaultMaskACT3.append(faultmaskact3)
                FaultMaskACT4.append(faultmaskact4)
                FaultMaskNEG.append(faultmaskneg)
            self.FaultMask = torch.stack(FaultMask)
            self.FaultMaskACT1 = torch.stack(FaultMaskACT1)
            self.FaultMaskACT2 = torch.stack(FaultMaskACT2)
            self.FaultMaskACT3 = torch.stack(FaultMaskACT3)
            self.FaultMaskACT4 = torch.stack(FaultMaskACT4)
            self.FaultMaskNEG = torch.stack(FaultMaskNEG)

    def RemoveFault(self):
        self.FaultMask = torch.ones_like(
            self.theta_).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT1 = torch.zeros(
            self.theta_.shape[1]).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT2 = torch.zeros(
            self.theta_.shape[1]).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT3 = torch.zeros(
            self.theta_.shape[1]).repeat(self.N_fault, 1, 1)
        self.FaultMaskACT4 = torch.zeros(
            self.theta_.shape[1]).repeat(self.N_fault, 1, 1)
        self.FaultMaskNEG = torch.zeros(
            self.theta_.shape[0]).repeat(self.N_fault, 1, 1)

    def UpdateArgs(self, args):
        self.args = args

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        self.INV.N = N
        self.INV.epsilon = epsilon


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

        # define nonlinear circuits
        # self.act = TanhRT(args)
        self.act1 = TanhRT(args)
        self.act2 = pReLURT(args)
        self.act3 = SigmoidRT(args)
        self.act4 = ClippedReLU(args)
        self.inv = InvRT(args)

        # area
        self.area_theta = torch.tensor(args.area_theta).to(self.device)
        # self.area_act = torch.tensor(args.area_act).to(self.device)
        # self.area_neg = torch.tensor(args.area_neg).to(self.device)

        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            # print('check topology: ', i, topology[i], topology[i+1])
            self.model.add_module(
                f'{i}-th Layer', pLayer(topology[i], topology[i+1], args, self.act1, self.act2, self.act3, self.act4, self.inv))

    def forward(self, x):
        self.RemoveFault()
        self.SampleFault()
        x = x.repeat(self.N_fault, self.N, 1, 1)
        out = self.model(x)
        # print("forward pass ----- ", x.shape, out.shape,
        #       self.args.N_fault, self.args.e_fault, self.N_fault, self.e_fault)
        # print(self.N_fault, self.e_fault)
        return out

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def soft_count_neg(self):
        soft_count = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_neg'):
                soft_count += l.soft_num_neg
        return soft_count

    @property
    def soft_count_act(self):
        soft_count1 = torch.tensor([0.]).to(self.device)
        soft_count2 = torch.tensor([0.]).to(self.device)
        soft_count3 = torch.tensor([0.]).to(self.device)
        soft_count4 = torch.tensor([0.]).to(self.device)
        for l in self.model:
            if hasattr(l, 'soft_num_act'):
                soft_count1 += l.soft_num_act[0]
                soft_count2 += l.soft_num_act[1]
                soft_count3 += l.soft_num_act[2]
                soft_count4 += l.soft_num_act[3]
        return soft_count1, soft_count2, soft_count3, soft_count4

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
        return self.inv.power * 1e-6 * self.soft_count_neg

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
        # print('---------- check power ----------',
        #       self.power_neg, self.power_act, self.power_mac)
        return self.power_neg + self.power_act + self.power_mac

    @property
    def Area(self):
        act_area = self.soft_count_act[0] * self.act1.Area + \
            self.soft_count_act[1] * self.act2.Area + \
            self.soft_count_act[2] * self.act3.Area + \
            self.soft_count_act[3] * self.act4.Area
        # print('check area: ', self.act1.Area)
        # print(self.inv.Area * self.soft_count_neg +
        #       self.area_theta * self.soft_count_theta)
        return self.inv.Area * self.soft_count_neg + act_area + self.area_theta * self.soft_count_theta
        # return 0

    @property
    def ActCount(self):
        return {self.act1.name: self.soft_count_act[0], self.act2.name: self.soft_count_act[1], self.act3.name: self.soft_count_act[2], self.act4.name: self.soft_count_act[3]}

    @property
    def DeviceCount(self):
        count_act = self.soft_count_act[0] + \
            self.soft_count_act[1] + self.soft_count_act[2] + \
            self.soft_count_act[3]
        return self.soft_count_neg + count_act + self.soft_count_theta

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
        self.args = args
        self.act1.args = args
        self.act2.args = args
        self.act3.args = args
        self.act4.args = args
        self.inv.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)

    def UpdateVariation(self, N, epsilon):
        self.N = N
        self.epsilon = epsilon
        self.act1.N = N
        self.act2.N = N
        self.act3.N = N
        self.act4.N = N
        self.act1.epsilon = epsilon
        self.act2.epsilon = epsilon
        self.act3.epsilon = epsilon
        self.act4.epsilon = epsilon
        self.inv.N = N
        self.inv.epsilon = epsilon
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

    def forward(self, y, label, power_ratio, power_consumption):
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
        loss += power_ratio * power_consumption.item()
        return loss
