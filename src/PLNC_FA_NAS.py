import numpy as np
import torch
import pandas as pd

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

        # non fault
        # /R1_open Y
        # /R1_short Y
        # /R2_open Y
        # /R2_short Y
        # /R3_open Y
        # /R3_short Y
        # /M1_G-S_short Y
        # /M1_open Y
        # /M1_G-D_short Y
        # /M1_D-S_short Y
        # /M2_D-S_short Y
        # /M2_G-D_short Y
        # /M2_G-S_short Y
        # /M2_open Y
        # /M3_open Y
        # /M3_D-S_short Y
        # /M3_G-D_short Y
        # /M3_G-S_short Y

        robust_eta_fault = torch.tensor(pd.read_csv(
            './Simulation/single_fault/robust/InvRT_eta_fault.csv').values)
        normal_eta_fault = torch.tensor([[-4.4411e-02, -9.5086e-01, -1.6900e-02, -1.4223e+02],
                                         [8.3907e-01, -1.0000e+00,
                                          0.0000e+00,  2.0978e-17],
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
                                         [-9.9992e-01, -1.0000e+00,  0.0000e+00, -4.5596e-18]])

        self.eta_fault = robust_eta_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else normal_eta_fault.to(self.DEVICE)

        self.robust_power_fault = torch.tensor(torch.load(
            './Simulation/single_fault/robust/power_inv_fault_normal.data').values)[0]
        self.normal_power_fault = torch.tensor([5.1250e+01, 3.9900e-03, 9.1230e+01, 8.4410e+01, 5.3410e+00, 5.1250e+01,
                                                5.1250e+01, 6.6555e+01, 9.1230e+01, 6.6550e+01, 1.1420e+01, 8.8681e-09,
                                                7.3398e+01, 1.1420e+01, 5.1900e+00, 1.0000e-15, 5.1245e+01, 6.5140e+01,
                                                1.0519e+02])

        self.power_fault = self.robust_power_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else self.normal_power_fault.to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'InvRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def DeviceCount(self):
        return 18 if self.args.type_nonlinear == 'robust' else 6

    @property
    def Area(self):
        return 8.17 if self.args.type_nonlinear == 'robust' else 4.062

    @property
    def normal_area(self):
        return 4.062

    @property
    def robust_area(self):
        return 8.17

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        # RTn_extend considered the variations in number N
        # so the dimension of RTn_extend is (N, 1, 1, 1) and then mean() to get the average power
        # but here as we have only one sample, we can directly use the power_estimator
        result = []
        for i in range(self.Mask.shape[0]):
            slices = []
            # m=0 means no fault
            # m is integer from 1
            mask = self.Mask[i].flatten()
            for j, m in enumerate(mask):
                power_temp = self.power_fault[int(m)].repeat(self.N, 1)
                slices.append(power_temp)

            output = torch.stack(slices, dim=2)

            result.append(output)
        power_out = torch.stack(result)
        return power_out.mean()

        # power_n = self.power_estimator(self.RTn_extend)
        # power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        # return power.mean()

    def output_variation(self, eta, z):
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = -(eta[i, 0] + eta[i, 1] *
                           torch.tanh((z[i, :, :] - eta[i, 2]) * eta[i, 3]))
        return a

    def output_faults(self, z, mask):
        slices = []
        # m=0 means no fault
        # m is integer from 1
        for i, m in enumerate(mask):
            eta_temp = self.eta_fault[int(m), :].repeat(self.N, 1)
            slices.append(self.output_variation(eta_temp, z)[:, :, i])

        output = torch.stack(slices, dim=2)
        return output

    def forward(self, z):
        result = []
        # iterate over different faults
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

        robust_eta_fault = torch.tensor(pd.read_csv(
            './Simulation/single_fault/robust/TanhRT_eta_fault.csv').values)
        normal_eta_fault = torch.tensor([[0.0568, -0.9188,  0.2366, -9.2469],
                                         [4.6647e-02, -9.5249e-01, -
                                          6.7710e-01,  1.2716e+01],
                                         [-9.0895e-01, -1.0020e+00, -
                                          1.1339e-02,  1.3834e-03],
                                         [-6.2602e-01, -2.0920e-02,
                                          6.0946e-01,  1.5644e+00],
                                         [9.9985e-01, -1.0000e+00,
                                          0.0000e+00,  2.4316e-17],
                                         [9.9985e-01, -1.0000e+00,
                                          0.0000e+00,  2.4316e-17],
                                         [-9.1036e-01, -1.0000e+00,
                                          0.0000e+00, -4.0192e-17],
                                         [1.0000e+00, -1.0000e+00,
                                          0.0000e+00,  4.4982e-16],
                                         [-9.1036e-01, -1.0000e+00,
                                          0.0000e+00, -4.0192e-17],
                                         [1.0000e+00, -1.0000e+00,
                                          0.0000e+00,  4.4982e-16],
                                         [9.9985e-01, -1.0000e+00,
                                          0.0000e+00,  2.4316e-17],
                                         [-1.0000e+00, -1.0000e+00,
                                          0.0000e+00,  4.4070e-16],
                                         [-1.2904e+01, -1.8619e+01,  1.9501e+00, -2.0736e+00]])

        self.eta_fault = robust_eta_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else normal_eta_fault.to(self.DEVICE)

        self.robust_power_fault = torch.tensor(torch.load(
            './Simulation/single_fault/robust/power_tanh_fault_normal.data').values)[0]
        self.normal_power_fault = torch.tensor([5.7330e+01, 6.6560e+01, 0.0000e+00, 1.0440e+02, 7.3090e+01, 1.1860e+00,
                                                4.0000e+00, 2.6740e+05, 9.1240e+01, 4.4460e+00, 1.1430e+01, 1.3850e+02,
                                                5.0320e+04]).to(self.DEVICE)

        self.power_fault = self.robust_power_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else self.normal_power_fault.to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'TanhRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def DeviceCount(self):
        return 12 if self.args.type_nonlinear == 'robust' else 4

    @property
    def Area(self):
        return 4.904 if self.args.type_nonlinear == 'robust' else 2.45

    @property
    def normal_area(self):
        return 2.45

    @property
    def robust_area(self):
        return 4.904

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        # RTn_extend considered the variations in number N
        # so the dimension of RTn_extend is (N, 1, 1, 1) and then mean() to get the average power
        # but here as we have only one sample, we can directly use the power_estimator
        # print(self.power_fault.shape, self.eta_fault.shape)
        result = []
        for i in range(self.Mask.shape[0]):
            slices = []
            # m=0 means no fault
            # m is integer from 1
            mask = self.Mask[i].flatten()
            for j, m in enumerate(mask):
                power_temp = self.power_fault[int(m)].repeat(self.N, 1)
                slices.append(power_temp)

            output = torch.stack(slices, dim=2)

            result.append(output)
        power_out = torch.stack(result)
        return power_out.mean()

    def output_variation(self, eta, z):
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = eta[i, 0] + eta[i, 1] * \
                torch.tanh((z[i, :, :] - eta[i, 2]) * eta[i, 3])
        return a

    def output_faults(self, z, mask):
        slices = []
        for i, m in enumerate(mask):
            eta_temp = self.eta_fault[int(m), :].repeat(self.N, 1)
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
# ======================================================  Learnable Clipped ReLU Activation  =====================================================
# ================================================================================================================================================


class ClippedReLU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        args.N_fault = args.N_fault

        robust_eta_fault = torch.tensor(pd.read_csv(
            './Simulation/single_fault/robust/ClippedReLU_eta_fault.csv').values)
        normal_eta_fault = torch.tensor([[0.0015,  0.9613, -0.0266,  1.1767],
                                         [2.2930e-03,  9.9901e-01, -
                                          4.6596e-01,  7.0913e-01],
                                         [6.0268e-07,  3.1167e-04,
                                          2.5654e-01,  1.7848e+00],
                                         [1.7028e-03,  9.7721e-01, -
                                          9.2145e-03,  1.1823e+00],
                                         [-2.0000e+00,  2.0003e+00, -
                                          2.0000e+00,  2.0003e+00],
                                         [1.0000e+00,  1.0000e+00,  2.0000e-01,  4.0000e-01]])

        self.eta_fault = robust_eta_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else normal_eta_fault.to(self.DEVICE)

        self.robust_power_fault = torch.tensor(torch.load(
            './Simulation/single_fault/robust/power_cp_fault_normal.data').values)[0]
        self.normal_power_fault = torch.tensor(
            [1.4520, 0.0353, 0.0177, 0.0000, 9.2810, 5.0830])

        self.power_fault = self.robust_power_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else self.normal_power_fault.to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'ClippedReLU'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def DeviceCount(self):
        return 6 if self.args.type_nonlinear == 'robust' else 2

    @property
    def Area(self):
        return 6.732 if self.args.type_nonlinear == 'robust' else 5.708

    @property
    def normal_area(self):
        return 5.708

    @property
    def robust_area(self):
        return 6.732

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        # RTn_extend considered the variations in number N
        # so the dimension of RTn_extend is (N, 1, 1, 1) and then mean() to get the average power
        # but here as we have only one sample, we can directly use the power_estimator
        result = []
        for i in range(self.Mask.shape[0]):
            slices = []
            # m=0 means no fault
            # m is integer from 1
            mask = self.Mask[i].flatten()
            for j, m in enumerate(mask):
                power_temp = self.power_fault[int(m)].repeat(self.N, 1)
                slices.append(power_temp)

            output = torch.stack(slices, dim=2)

            result.append(output)
        power_out = torch.stack(result)
        return power_out.mean()

    def output_variation(self, eta, z):
        a = torch.zeros_like(z)
        for i in range(self.N):
            linear_segment = eta[i, 0] + (eta[i, 1] - eta[i, 0]) / (
                eta[i, 3] - eta[i, 2]) * (z[i, :, :] - eta[i, 2])
            a[i, :, :] = torch.where(z[i, :, :] < eta[i, 2], eta[i, 0], torch.where(
                z[i, :, :] <= eta[i, 3], linear_segment, eta[i, 1]))
        return a

    def output_faults(self, z, mask):
        slices = []
        for i, m in enumerate(mask):
            eta_temp = self.eta_fault[int(m), :].repeat(self.N, 1)
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
# =======================================================  Learnable Soft pReLU Activation  ======================================================
# ================================================================================================================================================


class pReLURT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train

        # 'non_faulty',
        # 'R1_short',
        # 'R1_open',
        # 'R3_short',
        # 'R2_open',
        # 'R3_open',
        # 'R4_open',
        # 'R2_short',
        # 'M1_G-D short',
        # 'M1_G-S short',
        # 'M1_D-S short',
        # 'M1-open',
        # 'R4_short'

        robust_eta_fault = torch.tensor(pd.read_csv(
            './Simulation/single_fault/robust/pReLURT_eta_fault.csv').values)
        normal_eta_fault = torch.tensor([[1.8413e-02,  6.6951e-01,  2.0333e-01, -3.3020e-03,  7.7442e+00],
                                         [7.6165e-03,  6.8885e-01,
                                          1.8915e-01, -9.0097e-03,  7.6195e+00],
                                         [6.9539e-01, -6.9520e-01, -1.8925e-01,
                                          1.0383e-02,  7.6570e+00],
                                         [7.1250e-01, -5.3098e-01, -
                                          2.2832e-01, -3.2300e-02,  6.7483e+00],
                                         [-2.1845e-04,  6.9607e-01,
                                          1.8774e-01, -1.0552e-02,  7.6569e+00],
                                         [-2.1845e-04,  6.9607e-01,
                                          1.8774e-01, -1.0552e-02,  7.6569e+00],
                                         [-2.1845e-04,  6.9607e-01,
                                          1.8774e-01, -1.0552e-02,  7.6569e+00],
                                         [-2.1845e-04,  6.9607e-01,
                                          1.8774e-01, -1.0552e-02,  7.6569e+00],
                                         [7.6164e-03,  6.8885e-01,
                                          1.8915e-01, -9.0097e-03,  7.6195e+00],
                                         [7.1242e-01, -5.3090e-01, -
                                          2.2821e-01, -3.2292e-02,  6.7537e+00],
                                         [1.0003e+00, -1.5525e-06, -2.9949e-04,
                                          2.0097e-04,  9.9770e+00],
                                         [1.0003e+00, -1.5525e-06, -2.9949e-04,
                                          2.0097e-04,  9.9770e+00],
                                         [1.0003e+00, -1.5525e-06, -2.9949e-04,  2.0097e-04,  9.9770e+00]])

        self.eta_fault = robust_eta_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else normal_eta_fault.to(self.DEVICE)

        self.robust_power_fault = torch.tensor(torch.load(
            './Simulation/single_fault/robust/power_relu_fault_normal.data').values)[0]
        self.normal_power_fault = torch.tensor([30.8000,  31.3200,  30.8600,  15.8600,  30.8600,  15.1600,  15.6400,
                                                41.8400,  31.3200,  41.8400, 104.4000,  15.8400,   1.3340]).to(self.DEVICE)

        self.power_fault = self.robust_power_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else self.normal_power_fault.to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'pReLURT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def DeviceCount(self):
        return 12 if self.args.type_nonlinear == 'robust' else 5

    @property
    def Area(self):
        return 6.524 if self.args.type_nonlinear == 'robust' else 6.45401

    @property
    def normal_area(self):
        return 6.45401

    @property
    def robust_area(self):
        return 6.524

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        # RTn_extend considered the variations in number N
        # so the dimension of RTn_extend is (N, 1, 1, 1) and then mean() to get the average power
        # but here as we have only one sample, we can directly use the power_estimator
        result = []
        for i in range(self.Mask.shape[0]):
            slices = []
            # m=0 means no fault
            # m is integer from 1
            mask = self.Mask[i].flatten()
            for j, m in enumerate(mask):
                power_temp = self.power_fault[int(m)].repeat(self.N, 1)
                slices.append(power_temp)

            output = torch.stack(slices, dim=2)

            result.append(output)
        power_out = torch.stack(result)
        return power_out.mean()

    def output_variation(self, eta, z):
        def softplus(x, beta):
            return (1.0 / beta) * torch.log(1 + torch.exp(beta * x))
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = eta[i, 0] * (z[i, :, :] - eta[i, 2]) + eta[i, 1] * softplus(
                z[i, :, :] - eta[i, 2], eta[i, 4]) + eta[i, 3]
        return a

    def output_faults(self, z, mask):
        slices = []
        for i, m in enumerate(mask):
            eta_temp = self.eta_fault[int(m), :].repeat(self.N, 1)
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
# ========================================================  Learnable Sigmoid Activation  ========================================================
# ================================================================================================================================================


class SigmoidRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.N_train
        self.epsilon = args.e_train
        args.N_fault = args.N_fault

        robust_eta_fault = torch.tensor(pd.read_csv(
            './Simulation/single_fault/robust/SigmoidRT_eta_fault.csv').values)
        normal_eta_fault = torch.tensor([[0.9987,  -0.9195,   0.1099, -48.7125],
                                         [5.5417e-02,  9.3647e-01,
                                          1.4002e+00,  4.9676e+01],
                                         [-2.7483e+00,  2.7530e+00,
                                          1.4129e+00,  1.0269e+02],
                                         [5.3625e-03, -8.6046e-18,
                                          1.0000e+00,  5.0000e+01],
                                         [-3.0510e+01,  4.7455e+01,
                                          1.9509e+00,  9.8507e+00],
                                         [1.0000e+00,  1.1965e-16,
                                          1.0000e+00,  5.0000e+01],
                                         [9.9985e-01,  1.0039e-09,
                                          1.2932e+00,  5.6366e+01],
                                         [5.5417e-02,  9.3647e-01,
                                          1.4002e+00,  4.9676e+01],
                                         [9.9996e-01, -9.5299e-01,
                                          1.7045e+00,  1.9546e+01],
                                         [2.2500e-04,  3.4604e-19,
                                          1.0000e+00,  5.0000e+01],
                                         [-2.0000e+00, -2.3930e-16,
                                          1.0000e+00,  5.0000e+01],
                                         [2.5361e-03,  2.0784e-04,
                                          1.2791e+00,  1.0141e+01],
                                         [3.1594e-01, -2.5009e-02,
                                          1.5581e+00,  5.9755e+00],
                                         [-2.7500e+00, -1.5708e-15,
                                          1.0000e+00,  5.0000e+01],
                                         [4.2621e-05,  2.0892e-04,  1.2755e+00,  1.0126e+01]])

        self.eta_fault = robust_eta_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else normal_eta_fault.to(self.DEVICE)

        self.robust_power_fault = torch.tensor(torch.load(
            './Simulation/single_fault/robust/power_sigm_fault_normal.data').values)[0]
        self.normal_power_fault = torch.tensor([1.2810e+01, 2.2480e+01, 6.7220e+00, 1.3420e+04, 1.1270e+00, 9.9800e-04,
                                                2.8560e+00, 1.1980e+00, 2.2450e+01, 1.8830e+09, 1.2130e+00, 1.8830e+09,
                                                2.8560e+00, 1.2130e+00, 1.1990e+00])

        self.power_fault = self.robust_power_fault.to(
            self.DEVICE) if args.type_nonlinear == 'robust' else self.normal_power_fault.to(self.DEVICE)

        self.Mask = None

    @property
    def name(self):
        return 'SigmoidRT'

    @property
    def DEVICE(self):
        return self.args.DEVICE

    @property
    def DeviceCount(self):
        return 15 if self.args.type_nonlinear == 'robust' else 5

    @property
    def Area(self):
        return 7.396 if self.args.type_nonlinear == 'robust' else 2.458

    @property
    def normal_area(self):
        return 2.458

    @property
    def robust_area(self):
        return 7.396

    @property
    def power(self):
        # calculate power
        # returns a value not tensor!
        # RTn_extend considered the variations in number N
        # so the dimension of RTn_extend is (N, 1, 1, 1) and then mean() to get the average power
        # but here as we have only one sample, we can directly use the power_estimator
        result = []
        for i in range(self.Mask.shape[0]):
            slices = []
            # m=0 means no fault
            # m is integer from 1
            mask = self.Mask[i].flatten()
            for j, m in enumerate(mask):
                power_temp = self.power_fault[int(m)].repeat(self.N, 1)
                slices.append(power_temp)

            output = torch.stack(slices, dim=2)

            result.append(output)
        power_out = torch.stack(result)
        return power_out.mean()

    def output_variation(self, eta, z):
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i, :, :] = eta[i, 0] + eta[i, 1] * \
                torch.sigmoid((z[i, :, :] - eta[i, 2]) * eta[i, 3])
        return a

    def output_faults(self, z, mask):
        slices = []
        for i, m in enumerate(mask):
            eta_temp = self.eta_fault[int(m), :].repeat(self.N, 1)
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
