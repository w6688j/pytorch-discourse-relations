from __future__ import print_function

import argparse
import json
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.externals import joblib

import common
import util

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--conf', default='conf/default.json', help='path to conf file')
parsed_args = parser.parse_args()
args = json.load(open(parsed_args.conf))

# create vocab
print("===> creating vocabs ...")
end = time.time()
v_builder, d_word_index, embed, embedding_size, results_path = common.create_word_index(args['model'], args['glove'],
                                                                                        args['embedding_size'],
                                                                                        args['min_samples'],
                                                                                        args['pdtb_category'])

print('===> vocab creatin: {t:.3f}'.format(t=time.time() - end))
print('args: ', args)

# create trainer
print("===> creating dataloaders ...")
end = time.time()
train_loader = common.get_data_loader(args['model'], 'train', d_word_index, args['batch_size'],
                                      args['max_sentence_length'],
                                      pdtb_category=args['pdtb_category'])
val_loader = common.get_data_loader(args['model'], 'test', d_word_index, args['batch_size'],
                                    args['max_sentence_length'],
                                    pdtb_category=args['pdtb_category'])
print('===> dataloader creatin: {t:.3f}'.format(t=time.time() - end))

# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model, optimizer, criterion = common.get_model(model=args['model'],
                                               model_path=results_path,
                                               vocab_size=vocab_size,
                                               embedding_size=embedding_size,
                                               classes=args['classes'],
                                               rnn_model=args['rnn'],
                                               mean_seq=args['mean_seq'],
                                               hidden_size=args['hidden_size'],
                                               embed=embed,
                                               layers=args['layers'],
                                               lr=args['learning_rate'],
                                               weight_decay=args['weight_decay'],
                                               pdtb_category=args['pdtb_category'])
print(model)
print(optimizer)
print(criterion)

if args['cuda']:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()

# training and testing
start_time = time.time()
# 创建文件写控制器，将之后的数值以protocol buffer格式写入到logs文件夹中，空的logs文件夹将被自动创建。
writer = SummaryWriter()
for epoch in range(1, args['epochs'] + 1):
    util.adjust_learning_rate(args['learning_rate'], optimizer, epoch)
    if args['model'] == 'grn16':
        common.train_grn16(train_loader, model, criterion, optimizer, epoch, args['cuda'], args['clip'],
                           args['print_freq'])
        common.test_grn16(val_loader, model, criterion, args['cuda'], args['print_freq'])
    elif args['model'] == 'keann':
        common.train_keann(train_loader, model, criterion, optimizer, epoch, args['cuda'], args['clip'],
                           args['print_freq'])
        common.test_keann(val_loader, model, criterion, args['cuda'], args['print_freq'],args['pdtb_category'])
    elif args['model'] == 'keann_kg':
        common.train_keann_kg(train_loader, model, criterion, optimizer, epoch, args['cuda'], args['clip'],
                              args['print_freq'], writer)
        common.test_keann_kg(val_loader, model, criterion, args['cuda'], args['print_freq'],
                             args['pdtb_category'])
    else:
        common.train(train_loader, model, criterion, optimizer, epoch, args['cuda'], args['clip'], args['print_freq'])
        common.test(val_loader, model, criterion, args['cuda'], args['print_freq'])
        print('cost_time:%.4f' % (time.time() - start_time))
    # save current model
    if epoch % args['save_freq'] == 0:
        if args['pdtb_category']:
            path_save_model = results_path + '/' + args['model'] + '_' + args['pdtb_category'] + '_{}.pkl'.format(epoch)
        else:
            path_save_model = results_path + '/' + args['model'] + '_{}.pkl'.format(epoch)
        joblib.dump(model.float(), path_save_model, compress=2)
