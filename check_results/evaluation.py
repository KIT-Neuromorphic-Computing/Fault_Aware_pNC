import sys
import os
from pathlib import Path
import pickle
import torch
import pprint
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
from utils import *
from configuration import *

if not os.path.exists('./evaluation/'):
    os.makedirs('./evaluation/')

args = parser.parse_args()
args = FormulateArgs(args)

valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset/')
test_loader , datainfo = GetDataLoader(args, 'test',  path='../dataset/')
for x,y in valid_loader:
    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
for x,y in test_loader:
    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)

if not os.path.exists(f"./evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_eTrain_{args.e_train}_eTest{args.e_test}_fTrain_{args.e_fault}_fTest{args.e_fault_test}.matrix"):

    N_Faults = 500
    results = torch.zeros([N_Faults, 6])
    evaluator = Evaluator(args).to(args.DEVICE)
    modelname = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_epsilon_{args.e_train}_faults_{args.e_fault:1d}.model"

    if os.path.exists(f'./models/{modelname}'):
        model = torch.load(f'./models/{modelname}', map_location=args.DEVICE)
        print(modelname)
        model.UpdateArgs(args)
        
        model.UpdateVariation(args.N_test, args.e_test)
        model.UpdateFault(1, args.e_fault_test)
    
        SetSeed(args.SEED)

        for faultsample in range(N_Faults):
            print(faultsample)
        
            with torch.no_grad():
                result_valid = evaluator(model, X_valid, y_valid)
                valid_acc, _, valid_power, valid_area = result_valid['acc'], result_valid['std'], result_valid['power'], result_valid['area']
                result_test  = evaluator(model, X_test,  y_test)
                test_acc, _, test_power, test_area = result_test['acc'], result_test['std'], result_test['power'], result_test['area']
        
            results[faultsample, 0] = valid_acc
            results[faultsample, 1] = valid_power
            results[faultsample, 2] = valid_area
            results[faultsample, 3] = test_acc
            results[faultsample, 4] = test_power
            results[faultsample, 5] = test_area
                
                            
        torch.save(results, f"./evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_eTrain_{args.e_train}_eTest{args.e_test}_fTrain_{args.e_fault}_fTest{args.e_fault_test}.matrix")