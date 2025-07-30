import src.pNN_FA as pNN
from configuration import *
from utils import *
import sys
import os
from pathlib import Path
import pickle
import torch
import pprint
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))

if not os.path.exists('./FaultAnalysisRelu/evaluation/'):
    os.makedirs('./FaultAnalysisRelu/evaluation/')

args = parser.parse_args()
args = FormulateArgs(args)

valid_loader, datainfo = GetDataLoader(args, 'valid', path='./dataset/')
test_loader, datainfo = GetDataLoader(args, 'test',  path='./dataset/')
pprint.pprint(datainfo)

for x, y in valid_loader:
    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
for x, y in test_loader:
    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)


if not os.path.exists(f"./FaultAnalysisRelu/evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}.matrix"):

    N_Faults = 500
    e_faults = [0, 1, 2, 4]
    results = torch.zeros([N_Faults, 4, 2])

    topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
    pnn = pNN.pNN(topology, args).to(args.DEVICE)

    modelname = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{
        args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}.model"
    trained_model = torch.load(f'./FaultAnalysisRelu/models/{modelname}')
    trained_model.UpdateVariation(1, 0.)
    # trained_model.UpdateDropout(0.)

    for i, j in zip(trained_model.model, pnn.model):
        j.theta_.data = i.theta_.data

    pnn.UpdateVariation(1, 0.)
    for i, e_fault in enumerate(e_faults):
        pnn.UpdateFault(1, e_fault)

        for faultsample in range(N_Faults):
            print(e_fault, faultsample)
            pred_valid = pnn(X_valid)[0, 0, :, :]
            acc_valid = (torch.argmax(pred_valid, dim=1) ==
                         y_valid).sum() / y_valid.numel()

            pred_test = pnn(X_test)[0, 0, :, :]
            acc_test = (torch.argmax(pred_test, dim=1)
                        == y_test).sum() / y_test.numel()

            results[faultsample, i, 0] = acc_valid
            results[faultsample, i, 1] = acc_test

    torch.save(results, f"./FaultAnalysisRelu/evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{
               args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}.txt")
