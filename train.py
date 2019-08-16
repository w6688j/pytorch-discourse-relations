from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.externals import joblib

import common
import util

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=300, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=4, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--glove', default='glove/glove.6B.100d.txt', help='path to glove txt')
parser.add_argument('--model', default='rnn', help='model')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--max-sentence-length', type=int, default=256, help='default length of arg1:arg2')
parser.add_argument('--pdtb-category', default='Comparison',
                    choices=['Comparison', 'Contingency', 'Temporal', 'Expansion', ''],
                    help='PDTB category')
args = parser.parse_args()

# create vocab
print("===> creating vocabs ...")
end = time.time()
v_builder, d_word_index, embed, embedding_size, results_path = common.create_word_index(args.model, args.glove,
                                                                                        args.embedding_size,
                                                                                        args.min_samples)

print('===> vocab creatin: {t:.3f}'.format(t=time.time() - end))
print('args: ', args)

# create trainer
print("===> creating dataloaders ...")
end = time.time()
train_loader = common.get_data_loader(args.model, 'train', d_word_index, args.batch_size, args.max_sentence_length,
                                      pdtb_category=args.pdtb_category)
val_loader = common.get_data_loader(args.model, 'test', d_word_index, args.batch_size, args.max_sentence_length,
                                    pdtb_category=args.pdtb_category)
print('===> dataloader creatin: {t:.3f}'.format(t=time.time() - end))

# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model, optimizer, criterion = common.get_model(model=args.model,
                                               model_path=results_path,
                                               vocab_size=vocab_size,
                                               embedding_size=embedding_size,
                                               classes=args.classes,
                                               rnn_model=args.rnn,
                                               mean_seq=args.mean_seq,
                                               hidden_size=args.hidden_size,
                                               embed=embed,
                                               layers=args.layers,
                                               lr=args.lr,
                                               weight_decay=args.weight_decay,
                                               pdtb_category=args.pdtb_category)
print(model)
print(optimizer)
print(criterion)

if args.cuda:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()

# training and testing
for epoch in range(1, args.epochs + 1):
    util.adjust_learning_rate(args.lr, optimizer, epoch)
    if args.model == 'grn16':
        common.train_grn16(train_loader, model, criterion, optimizer, epoch, args.cuda, args.clip, args.print_freq)
        common.test_grn16(val_loader, model, criterion, args.cuda, args.print_freq)
    elif args.model == 'keann':
        common.train_keann(train_loader, model, criterion, optimizer, epoch, args.cuda, args.clip, args.print_freq)
        common.test_keann(val_loader, model, criterion, args.cuda, args.print_freq)
    elif args.model == 'keann_kg':
        common.train_keann_kg(train_loader, model, criterion, optimizer, epoch, args.cuda, args.clip, args.print_freq)
        common.test_keann_kg(val_loader, model, criterion, args.cuda, args.print_freq)
    else:
        common.train(train_loader, model, criterion, optimizer, epoch, args.cuda, args.clip, args.print_freq)
        common.test(val_loader, model, criterion, args.cuda, args.print_freq)
    # save current model
    if epoch % args.save_freq == 0:
        if args.pdtb_category:
            path_save_model = results_path + '/' + args.model + '_' + args.pdtb_category + '_{}.pkl'.format(epoch)
        else:
            path_save_model = results_path + '/' + args.model + '_{}.pkl'.format(epoch)
        joblib.dump(model.float(), path_save_model, compress=2)
