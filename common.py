import gc
import os
import time

import torch
from sklearn.externals import joblib
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import dataloader
import models
import util
from vocab import VocabBuilder, GloveVocabBuilder

PROCESSED_DATA_PATH = 'data/processed_data'

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


def create_word_index(model, glove_path, embedding_size, min_samples, pdtb_category=''):
    if os.path.exists(glove_path):
        v_builder = GloveVocabBuilder(path_glove=glove_path)
        d_word_index, embed = v_builder.get_word_index()
        ed_size = embed.size(1)
        is_glove = True
    else:
        v_builder = VocabBuilder(path_file=PROCESSED_DATA_PATH + '/train.tsv')
        d_word_index, embed = v_builder.get_word_index(min_sample=min_samples)
        ed_size = embedding_size
        is_glove = False

    results_path = get_results_path(model, is_glove, ed_size, pdtb_category)
    joblib.dump(d_word_index, results_path + '/d_word_index.pkl', compress=3)

    return (v_builder, d_word_index, embed, ed_size, results_path)


def get_word_index(model, glove_path, embedding_size):
    if os.path.exists(glove_path):
        v_builder = GloveVocabBuilder(path_glove=glove_path)
        d_word_index, embed = v_builder.get_word_index()
        ed_size = embed.size(1)
        is_glove = True
    else:
        ed_size = embedding_size
        is_glove = False

    d_word_index = None
    results_path = get_results_path(model, is_glove, ed_size)
    if os.path.exists(results_path + '/d_word_index.pkl'):
        d_word_index = joblib.load(results_path + '/d_word_index.pkl')

    return d_word_index, results_path


def get_results_path(model, is_glove, embedding_size, pdtb_category=''):
    if is_glove:
        results_path = 'results/' + model + '_' + 'glove_' + str(embedding_size) + 'v' + pdtb_category
    else:
        results_path = 'results/' + model + '_' + 'no_glove_' + str(embedding_size) + 'v' + pdtb_category
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path


def get_data_loader(model, type, d_word_index, batch_size, max_length, pdtb_category):
    if type == 'train':
        path = PROCESSED_DATA_PATH + '/train.tsv'
    else:
        path = PROCESSED_DATA_PATH + '/test.tsv'

    if model == 'rnn':
        return dataloader.RnnTextClassDataLoader(path, d_word_index, batch_size=batch_size)
    elif model == 'rnnatt17':
        return dataloader.RnnAtt17TextClassDataLoader(path, d_word_index, batch_size=batch_size, max_length=max_length)
    elif model == 'grn16':
        if type == 'train':
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/train.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/train.tsv'
        else:
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/test.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/test.tsv'

        return dataloader.Grn16TextClassDataLoader(path, d_word_index, batch_size=batch_size, max_length=max_length)
    elif model == 'keann':
        if type == 'train':
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/train.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/train.tsv'
        else:
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/test.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/test.tsv'

        return dataloader.KeannTextClassDataLoader(path,
                                                   d_word_index,
                                                   batch_size=batch_size)
    elif model == 'keann_kg':
        if type == 'train':
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/train.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/train.tsv'
        else:
            if pdtb_category:
                path = PROCESSED_DATA_PATH + '/grn16/' + pdtb_category + '/test.tsv'
            else:
                path = PROCESSED_DATA_PATH + '/grn16/test.tsv'
        return dataloader.KeannKGTextClassDataLoader(path,
                                                     d_word_index,
                                                     batch_size=batch_size,
                                                     transE_model_path='transE/model/WN11/l_0.001_es_0_L_1_em_300_nb_100_n_100_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt',
                                                     transE_embedding_path='transE/model/WN11/ent_embeddings.txt',
                                                     transE_entity_path='transE/datasets/WN11/new_entity2id.txt')
    elif model == 'trans_s':
        if type == 'train':
            path = PROCESSED_DATA_PATH + '/grn16/train.tsv'
        else:
            path = PROCESSED_DATA_PATH + '/grn16/test.tsv'

        return dataloader.TransSTextClassDataLoader(path, d_word_index, batch_size=batch_size)


def get_model(model,
              model_path,
              pdtb_category,
              vocab_size=10000,
              embedding_size=50,
              classes=4,
              rnn_model='LSTM',
              mean_seq=False,
              hidden_size=128,
              embed=None,
              layers=2,
              lr=0.005,
              weight_decay=1e-4):
    if pdtb_category:
        model_file_path = model_path + '/' + model + '_' + pdtb_category + '_50.pkl'
    else:
        model_file_path = model_path + '/' + model + '_50.pkl'

    if model == 'rnn':
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.RNN(vocab_size=vocab_size,
                                embed_size=embedding_size,
                                num_output=classes,
                                rnn_model=rnn_model,
                                use_last=(not mean_seq),
                                hidden_size=hidden_size,
                                embedding_tensor=embed,
                                num_layers=layers,
                                batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)

    elif model == 'rnnatt17':
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.RNNATT17(vocab_size=vocab_size,
                                     embed_size=embedding_size,
                                     num_output=classes,
                                     rnn_model=rnn_model,
                                     use_last=(not mean_seq),
                                     hidden_size=hidden_size,
                                     embedding_tensor=embed,
                                     num_layers=layers,
                                     batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)

    elif model == 'grn16':
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.GRN16(vocab_size=vocab_size,
                                  embed_size=embedding_size,
                                  num_output=classes,
                                  rnn_model=rnn_model,
                                  use_last=(not mean_seq),
                                  hidden_size=hidden_size,
                                  embedding_tensor=embed,
                                  num_layers=layers,
                                  batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)

    elif model == 'keann':
        if pdtb_category:
            model_file_path = model_path + '/' + model + '_' + pdtb_category + '_10.pkl'
        else:
            model_file_path = model_path + '/' + model + '_10.pkl'
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.KEANN(vocab_size=vocab_size,
                                  embed_size=embedding_size,
                                  num_output=classes,
                                  rnn_model=rnn_model,
                                  use_last=(not mean_seq),
                                  hidden_size=hidden_size,
                                  embedding_tensor=embed,
                                  num_layers=layers,
                                  batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)

    elif model == 'keann_kg':
        if pdtb_category:
            model_file_path = model_path + '/' + model + '_' + pdtb_category + '_10.pkl'
        else:
            model_file_path = model_path + '/' + model + '_10.pkl'
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.KEANNKG(vocab_size=vocab_size,
                                    embed_size=embedding_size,
                                    num_output=classes,
                                    rnn_model=rnn_model,
                                    use_last=(not mean_seq),
                                    hidden_size=hidden_size,
                                    embedding_tensor=embed,
                                    num_layers=layers,
                                    batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)

    elif model == 'trans_s':
        # Todo
        if os.path.exists(model_file_path):
            _model = joblib.load(model_file_path)
        else:
            _model = models.KEANNKG(vocab_size=vocab_size,
                                    embed_size=embedding_size,
                                    num_output=classes,
                                    rnn_model=rnn_model,
                                    use_last=(not mean_seq),
                                    hidden_size=hidden_size,
                                    embedding_tensor=embed,
                                    num_layers=layers,
                                    batch_first=True)

        # optimizer and loss
        _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

        _criterion = nn.CrossEntropyLoss()

        return (_model, _optimizer, _criterion)


def train(train_loader, model, criterion, optimizer, epoch, cuda, clip, print_freq):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input, seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def test(val_loader, model, criterion, cuda, print_freq):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    prfa = util.AverageMeterPRFA()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(val_loader):

        if cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input, seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prfa_all = util.prf_multi_classify(output.data, target, topk=(1,))
        prfa.update(prfa_all, seq_lengths.size(0))
        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        four_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa)

    four_classify_last_print(top1, prfa)
    return top1.avg


def train_grn16(train_loader, model, criterion, optimizer, epoch, cuda, clip, print_freq):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (arg1, arg2, target, seq_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def test_grn16(val_loader, model, criterion, cuda, print_freq):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    prfa = util.AverageMeterPRFA()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (arg1, arg2, target, seq_lengths) in enumerate(val_loader):

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prfa_all = util.prf_multi_classify(output.data, target, topk=(1,))
        prfa.update(prfa_all, seq_lengths.size(0))

        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        four_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa)

    four_classify_last_print(top1, prfa)
    return top1.avg


def train_keann(train_loader, model, criterion, optimizer, epoch, cuda, clip, print_freq):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (arg1, arg2, target, seq_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def test_keann(val_loader, model, criterion, cuda, print_freq, pdtb_category=''):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    prfa = util.AverageMeterPRFA()
    if pdtb_category != '':
        prfa = util.AverageBinaryMeterPRFA()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (arg1, arg2, target, seq_lengths) in enumerate(val_loader):

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prfa_all = util.prf_multi_classify(output.data, target, topk=(1,))
        prfa.update(prfa_all, seq_lengths.size(0))

        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if pdtb_category != '':
            binary_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa, pdtb_category)
        else:
            four_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa)

    if pdtb_category != '':
        binary_classify_last_print(top1, prfa)
    else:
        four_classify_last_print(top1, prfa)
    return top1.avg


def transToTorch(sentence_list):
    seq_lengths = longTensor(list(map(len, sentence_list)))
    tensor_list = torch.zeros((len(sentence_list), 80)).long()

    for idx, (seq, seqlen) in enumerate(zip(sentence_list, seq_lengths)):
        if seqlen > 80:
            tensor_list[idx, :80] = longTensor(seq[:80])
        else:
            tensor_list[idx, :seqlen] = longTensor(seq)

    if USE_CUDA:
        tensor_list = tensor_list.cuda()

    return tensor_list


def train_keann_kg(train_loader, model, criterion, optimizer, epoch, cuda, clip, print_freq, writer):
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (arg1, arg2, target, seq_lengths, transE_tensor_arg1, transE_tensor_arg2, _) in enumerate(
            train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        arg1 = Variable(arg1, requires_grad=False)
        arg2 = Variable(arg2, requires_grad=False)
        target = Variable(target, requires_grad=False)

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths, (transE_tensor_arg1, transE_tensor_arg2), cuda)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.cpu().numpy(), seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1))
            gc.collect()

        # ============================== 画图 ===============================
        # 写入vae模型的损失值，可以在一张图上显示出来。
        # 注意 pytorch0.4 要在这个tensor后面加 .item()，这样存的数值不是计算图
        # epoch是图的x轴
        writer.add_scalars('data/train_loss', {'class Loss': loss.item()}, epoch)
        # 转为图像
        rand_matrix = vutils.make_grid(model.rand_matrix, normalize=True, scale_each=True)
        # 写入writer
        writer.add_image('Image', rand_matrix, epoch)

        # 转为图像
        kg_relation = vutils.make_grid(model.kg_relation, normalize=True, scale_each=True)
        # 写入writer
        writer.add_image('Image', kg_relation, epoch)

        # 转为图像
        last_tensor = vutils.make_grid(model.last_tensor, normalize=True, scale_each=True)
        # 写入writer
        writer.add_image('Image', last_tensor, epoch)


def test_keann_kg(val_loader, model, criterion, cuda, print_freq, pdtb_category=''):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    prfa = util.AverageMeterPRFA()
    if pdtb_category != '':
        prfa = util.AverageBinaryMeterPRFA()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (arg1, arg2, target, seq_lengths, transE_tensor_arg1, transE_tensor_arg2, batch_original) in enumerate(
            val_loader):

        arg1 = Variable(arg1, requires_grad=False)
        arg2 = Variable(arg2, requires_grad=False)
        target = Variable(target, requires_grad=False)

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths, (transE_tensor_arg1, transE_tensor_arg2), cuda)
        # hot_image(model.encoder(arg1), model.encoder(arg2), batch_original, model.rand_matrix, model.kg_relation)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prfa_all = util.prf_multi_classify(output.data, target, topk=(1,))
        prfa.update(prfa_all, seq_lengths.size(0))

        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.cpu().numpy(), seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if pdtb_category != '':
            binary_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa, pdtb_category)
        else:
            four_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa)

    if pdtb_category != '':
        binary_classify_last_print(top1, prfa)
    else:
        four_classify_last_print(top1, prfa)
    return top1.avg


def test_keann_kg2(val_loader, model, criterion, cuda, print_freq):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    prfa = util.AverageMeterPRFA()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (arg1, arg2, target, seq_lengths, transE_tensor_arg1, transE_tensor_arg2, batch_original) in enumerate(
            val_loader):

        arg1 = Variable(arg1, requires_grad=False)
        arg2 = Variable(arg2, requires_grad=False)
        target = Variable(target, requires_grad=False)

        if cuda:
            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            target = target.cuda()

        # compute output
        output = model((arg1, arg2), seq_lengths, (transE_tensor_arg1, transE_tensor_arg2), cuda)
        hot_image(model.encoder(arg1), model.encoder(arg2), batch_original, model.last_tensor, model.kg_relation_list)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prfa_all = util.prf_multi_classify(output.data, target, topk=(1,))
        prfa.update(prfa_all, seq_lengths.size(0))

        prec1 = util.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.cpu().numpy(), seq_lengths.size(0))
        top1.update(prec1[0][0], seq_lengths.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        four_classify_loop_print(i, print_freq, val_loader, batch_time, losses, top1, prfa)

    four_classify_last_print(top1, prfa)
    return top1.avg


# 画图
def hot_image(arg1_embs, arg2_embs, batch_original, last_tensor, kg_relation):
    arg1 = batch_original[0][1]
    arg2 = batch_original[0][2]
    arg1_emb = arg1_embs[0]
    arg2_emb = arg2_embs[0]
    word_num1 = len(arg1) if len(arg1) <= 80 else 80
    word_num2 = len(arg2) if len(arg2) <= 80 else 80

    # 画args cos相似度热图
    draw_hot_map_cos(arg1, arg2, arg1_emb, arg2_emb, word_num1=word_num1, word_num2=word_num2,
                     img_name='images/args.png')

    # 画last_tensor热图
    draw_hot_map_last_tensor(last_tensor[:word_num1, :word_num2], arg1, arg2)

    # 画kg_relation热图
    # draw_hot_map_kg_relation(kg_relation[0][:word_num2, :word_num1], arg1, arg2)
    exit()


# args cos相似度热图
def draw_hot_map_cos(x_list, y_list, x_embs, y_embs, word_num1=80, word_num2=80, img_name='hot-map.png'):
    similar_matrix = np.zeros((word_num1, word_num2))
    for i, word_1 in enumerate(x_list):
        if i >= word_num1:
            break
        for j, word_2 in enumerate(y_list):
            if j >= word_num2:
                break
            vec_1 = np.array([x_embs[i].cpu().detach().numpy()])
            vec_2 = np.array([y_embs[j].cpu().detach().numpy()])
            cos = cosine_similarity(vec_1, vec_2)[0, 0]
            similar_matrix[i][j] = cos

    save_matrix('images/similar_matrix.txt', similar_matrix)

    ax = sns.heatmap(similar_matrix, fmt="d")
    label_x = ax.set_xticklabels(y_list)
    label_y = ax.set_yticklabels(x_list)

    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    plt.savefig(img_name)
    plt.close()


def draw_hot_map_last_tensor(last_tensor, arg1, arg2):
    save_matrix('images/last_tensor.txt', last_tensor.cpu().detach().numpy())
    # 画last_tensor热图
    ax = sns.heatmap(last_tensor.cpu().detach().numpy(), fmt="d")
    label_x = ax.set_xticklabels(arg2)
    label_y = ax.set_yticklabels(arg1)
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    plt.savefig('images/last_tensor.png')
    plt.close()


def draw_hot_map_kg_relation(kg_relation, arg1, arg2):
    # 画kg_relation热图
    ax = sns.heatmap(np.fabs(kg_relation.cpu().detach().numpy()) * 1000, fmt="d", cmap='YlGnBu')
    label_x = ax.set_xticklabels(arg1)
    label_y = ax.set_yticklabels(arg2)
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    plt.savefig('images/kg_relation.png')
    plt.close()


def save_matrix(file_name, matrix):
    with open(file_name, 'a+', encoding='utf-8') as f:
        for item in matrix:
            f.write('\t'.join('%s' % id for id in item) + '\n')


def four_classify_loop_print(i,
                             print_freq,
                             val_loader,
                             batch_time,
                             losses,
                             top1,
                             prfa):
    if i != 0 and i % print_freq == 0:
        print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
              'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@ {top1.val:.3f} ({top1.avg:.3f}) \n'

              '     Comparison: Precision {prfa.Comparison_Precision.val:.4f} ({prfa.Comparison_Precision.avg:.4f})  '
              'Recall {prfa.Comparison_Recall.val:.4f} ({prfa.Comparison_Recall.avg:.4f})  '
              'F1 {prfa.Comparison_F1.val:.4f} ({prfa.Comparison_F1.avg:.4f})  \n'

              '     Contingency: Precision {prfa.Contingency_Precision.val:.4f} ({prfa.Contingency_Precision.avg:.4f})  '
              'Recall {prfa.Contingency_Recall.val:.4f} ({prfa.Contingency_Recall.avg:.4f})  '
              'F1 {prfa.Contingency_F1.val:.4f} ({prfa.Contingency_F1.avg:.4f})   \n'

              '     Temporal: Precision {prfa.Temporal_Precision.val:.4f} ({prfa.Temporal_Precision.avg:.4f})  '
              'Recall {prfa.Temporal_Recall.val:.4f} ({prfa.Temporal_Recall.avg:.4f})  '
              'F1 {prfa.Temporal_F1.val:.4f} ({prfa.Temporal_F1.avg:.4f})   \n'

              '     Expansion: Precision {prfa.Expansion_Precision.val:.4f} ({prfa.Expansion_Precision.avg:.4f})  '
              'Recall {prfa.Expansion_Recall.val:.4f} ({prfa.Expansion_Recall.avg:.4f})  '
              'F1 {prfa.Expansion_F1.val:.4f} ({prfa.Expansion_F1.avg:.4f})   \n'

              .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, prfa=prfa))
        gc.collect()


def four_classify_last_print(top1, prfa):
    print(' * Acc@ {top1.avg:.3f}  \n'
          ' * Comparison: Precision@ {prfa.Comparison_Precision.avg:.4f}  '
          'Recall@ {prfa.Comparison_Recall.avg:.4f}  '
          'F1@ {prfa.Comparison_F1.avg:.4f}   \n'

          ' * Contingency: Precision@ {prfa.Contingency_Precision.avg:.4f}  '
          'Recall@ {prfa.Contingency_Recall.avg:.4f}  '
          'F1@ {prfa.Contingency_F1.avg:.4f}   \n'

          ' * Temporal: Precision@ {prfa.Temporal_Precision.avg:.4f}  '
          'Recall@ {prfa.Temporal_Recall.avg:.4f}  '
          'F1@ {prfa.Temporal_F1.avg:.4f}   \n'

          ' * Expansion: Precision@ {prfa.Expansion_Precision.avg:.4f}  '
          'Recall@ {prfa.Expansion_Recall.avg:.4f}  '
          'F1@ {prfa.Expansion_F1.avg:.4f}   \n'

          .format(top1=top1, prfa=prfa))


def binary_classify_loop_print(i,
                               print_freq,
                               val_loader,
                               batch_time,
                               losses,
                               top1,
                               prfa,
                               binary_name='Comparison'):
    if i != 0 and i % print_freq == 0:
        print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
              'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@ {top1.val:.3f} ({top1.avg:.3f}) \n'

              ' * {binary_name}: Precision@ {prfa.Binary_Precision.avg:.4f}  '
              'Recall@ {prfa.Binary_Recall.avg:.4f}  '
              'F1@ {prfa.Binary_F1.avg:.4f}   \n'

              ' * Others: Precision@ {prfa.Others_Precision.avg:.4f}  '
              'Recall@ {prfa.Others_Recall.avg:.4f}  '
              'F1@ {prfa.Others_F1.avg:.4f}   \n'

              .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, prfa=prfa,
                      binary_name=binary_name))
        gc.collect()


def binary_classify_last_print(accCounter, prfa, binary_name='Comparison'):
    print(' * Acc@ {top1.avg:.4f}  \n'
          ' * {binary_name}: Precision@ {prfa.Binary_Precision.avg:.4f}  '
          'Recall@ {prfa.Binary_Recall.avg:.4f}  '
          'F1@ {prfa.Binary_F1.avg:.4f}   \n'

          ' * Others: Precision@ {prfa.Others_Precision.avg:.4f}  '
          'Recall@ {prfa.Others_Recall.avg:.4f}  '
          'F1@ {prfa.Others_F1.avg:.4f}   \n'
          .format(top1=accCounter, binary_name=binary_name, prfa=prfa))
