from utils import *
from src import pNN_FA_MIX_MATCH as pNN
import pprint
import torch
from configuration import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))
# import pNN

args = parser.parse_args()
args = FormulateArgs(args)

print(f'Training network on device: {args.DEVICE}.')
MakeFolder(args)

train_loader, datainfo = GetDataLoader(args, 'train')
valid_loader, datainfo = GetDataLoader(args, 'valid')
test_loader, datainfo = GetDataLoader(args, 'test')
pprint.pprint(datainfo)

SetSeed(args.SEED)

setup = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_epsilon_{
    args.e_train}_dropout_{args.dropout}_fault_ratio_{args.fault_ratio}.model"
print(f'Training setup: {setup}.')

msglogger = GetMessageLogger(args, setup)
msglogger.info(f'Training network on device: {args.DEVICE}.')
msglogger.info(f'Training setup: {setup}.')
msglogger.info(datainfo)

# args.e_fault = args.fault_ratio * (args.hidden[0] + datainfo['N_class'])
# print('Fault ratio:', args.fault_ratio, args.e_fault)

if os.path.isfile(f'{args.savepath}/{setup}'):
    print(f'{setup} exists, skip this training.')
    msglogger.info('Training was already finished.')
else:
    topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
    msglogger.info(f'Topology of the network: {topology}.')

    # print(args.N_fault, args.e_fault)
    pnn = pNN.pNN(topology, args).to(args.DEVICE)
    args.e_fault = int(pnn.DeviceCount1 * args.fault_ratio)
    if args.e_fault == 0:
        # Monte Carlo simulation
        args.N_fault = 1
    pnn.UpdateFault(N_fault=args.N_fault, e_fault=args.e_fault)
    print('Fault ratio:', args.fault_ratio,
          pnn.DeviceCount1, pnn.args.e_fault, pnn.args.N_fault, pnn.e_fault, pnn.N_fault)

    # print('Average power:', ave_power, 'Average area:', ave_area)
    lossfunction = pNN.pNNLoss(args).to(args.DEVICE)
    # print(args.N_fault, args.e_fault)
    optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR,)

    if args.PROGRESSIVE:
        pnn, best = train_pnn_progressive(
            pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        pnn, best = train_pnn(pnn, train_loader, valid_loader,
                              lossfunction, optimizer, args, msglogger, UUID=setup)

    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(pnn, f'{args.savepath}/{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')

CloseLogger(msglogger)
