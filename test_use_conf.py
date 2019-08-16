from __future__ import print_function

import argparse
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import common

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--conf', default='conf/default.json', help='path to conf file')
parsed_args = parser.parse_args()
args = json.load(open(parsed_args.conf))

d_word_index, results_path = common.get_word_index(args['model'], args['glove'], args['embedding_size'])

# create tester
print("===> creating dataloaders ...")
val_loader = common.get_data_loader(args['model'], 'test', d_word_index, args['batch_size'],
                                    args['max_sentence_length'],
                                    pdtb_category=args['pdtb_category'])

# load model,optimizer and loss
model, optimizer, criterion = common.get_model(model=args['model'],
                                               model_path=results_path,
                                               lr=args['learning_rate'],
                                               weight_decay=args['weight_decay'],
                                               pdtb_category=args['pdtb_category'])
print(optimizer)
print(criterion)

if args['cuda']:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()

if args['model'] == 'grn16':
    common.test_grn16(val_loader, model, criterion, args['cuda'], args['print_freq'])
elif args['model'] == 'keann':
    common.test_keann(val_loader, model, criterion, args['cuda'], args['print_freq'])
elif args['model'] == 'keann_kg':
    common.test_keann_kg(val_loader, model, criterion, args['cuda'], args['print_freq'])
else:
    common.test(val_loader, model, criterion, args['cuda'], args['print_freq'])
