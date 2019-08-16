import csv
import os

PDTB_LABEL_TO_INDEX = {
    'Comparison': 0,
    'Contingency': 1,
    'Temporal': 2,
    'Expansion': 3
}

PDTB_LABEL_TO_INDEX_TYPE = {
    'Comparison': {
        'Comparison': 1,
        'Contingency': 0,
        'Temporal': 0,
        'Expansion': 0
    },
    'Contingency': {
        'Comparison': 0,
        'Contingency': 1,
        'Temporal': 0,
        'Expansion': 0
    },
    'Temporal': {
        'Comparison': 0,
        'Contingency': 0,
        'Temporal': 1,
        'Expansion': 0
    },
    'Expansion': {
        'Comparison': 0,
        'Contingency': 0,
        'Temporal': 0,
        'Expansion': 1
    }
}


def create_pdtb_tsv_file(path_in, path_out):
    path = '/'.join(path_out.split('/')[0:-1])
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_in, 'r') as f, open(path_out, 'w') as fw:
        writer = csv.writer(fw, delimiter='\t')
        writer.writerow(['label', 'body'])
        for line in f:
            line_sp = line.strip().replace(',', ' ')
            line_sp = line_sp.replace('"', ' ')
            line_sp = line_sp.split('|||')
            arg_pairs = line_sp[1] + ' ' + line_sp[2]
            tokens = [x.lower() for x in arg_pairs.split()]
            label = PDTB_LABEL_TO_INDEX[line_sp[0]]
            body = ' '.join(tokens)
            writer.writerow([label, body])


def create_pdtb_tsv_file_type_rnnatt17(path_in, path_out, file_name):
    for type in PDTB_LABEL_TO_INDEX:
        path = path_out + '/rnnatt17/' + type
        file_path = path + '/' + file_name
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path_in, 'r') as f, open(file_path, 'w') as fw:
            writer = csv.writer(fw, delimiter='\t')
            writer.writerow(['label', 'body'])
            for line in f:
                line_sp = line.strip().replace(',', ' ')
                line_sp = line_sp.replace('"', ' ')
                line_sp = line_sp.split('|||')
                arg_pairs = line_sp[1] + ' ' + line_sp[2]
                tokens = [x.lower() for x in arg_pairs.split()]
                label = PDTB_LABEL_TO_INDEX_TYPE[type][line_sp[0]]
                body = ' '.join(tokens)
                writer.writerow([label, body])


def create_pdtb_tsv_file_grn16(path_in, path_out, file_name):
    path = path_out + '/grn16/'
    file_path = path + '/' + file_name
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path_in, 'r') as f, open(file_path, 'w') as fw:
        writer = csv.writer(fw, delimiter='\t')
        writer.writerow(['label', 'arg1', 'arg2'])
        for line in f:
            line_sp = line.strip().replace(',', ' ')
            line_sp = line_sp.replace('"', ' ')
            line_sp = line_sp.split('|||')
            tokens1 = [x.lower() for x in line_sp[1].split()]
            tokens2 = [x.lower() for x in line_sp[2].split()]
            label = PDTB_LABEL_TO_INDEX[line_sp[0]]
            arg1 = ' '.join(tokens1)
            arg2 = ' '.join(tokens2)
            writer.writerow([label, arg1, arg2])


def create_pdtb_tsv_file_type_grn16(path_in, path_out, file_name):
    for type in PDTB_LABEL_TO_INDEX:
        path = path_out + '/grn16/' + type
        file_path = path + '/' + file_name
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path_in, 'r') as f, open(file_path, 'w') as fw:
            writer = csv.writer(fw, delimiter='\t')
            writer.writerow(['label', 'arg1', 'arg2'])
            for line in f:
                line_sp = line.strip().replace(',', ' ')
                line_sp = line_sp.replace('"', ' ')
                line_sp = line_sp.split('|||')
                tokens1 = [x.lower() for x in line_sp[1].split()]
                tokens2 = [x.lower() for x in line_sp[2].split()]
                label = PDTB_LABEL_TO_INDEX_TYPE[type][line_sp[0]]
                arg1 = ' '.join(tokens1)
                arg2 = ' '.join(tokens2)
                writer.writerow([label, arg1, arg2])


def _tokenize(text):
    if not isinstance(text, str):
        return []
    else:
        # return [x.lower() for x in nltk.word_tokenize(text)]
        return [x.lower() for x in text.split()]


''' from https://github.com/pytorch/examples/blob/master/imagenet/main.py'''


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterPRFA():
    def __init__(self):
        self.Comparison_Precision = AverageMeter()
        self.Comparison_Recall = AverageMeter()
        self.Comparison_F1 = AverageMeter()
        self.Comparison_Accuracy = AverageMeter()

        self.Contingency_Precision = AverageMeter()
        self.Contingency_Recall = AverageMeter()
        self.Contingency_F1 = AverageMeter()
        self.Contingency_Accuracy = AverageMeter()

        self.Temporal_Precision = AverageMeter()
        self.Temporal_Recall = AverageMeter()
        self.Temporal_F1 = AverageMeter()
        self.Temporal_Accuracy = AverageMeter()

        self.Expansion_Precision = AverageMeter()
        self.Expansion_Recall = AverageMeter()
        self.Expansion_F1 = AverageMeter()
        self.Expansion_Accuracy = AverageMeter()

    def update(self, val, n=1):
        self.Comparison_Precision.update(val[0]['Precision'], n)
        self.Comparison_Recall.update(val[0]['Recall'], n)
        self.Comparison_F1.update(val[0]['F1_Score'], n)
        self.Comparison_Accuracy.update(val[0]['Accuracy'], n)

        self.Contingency_Precision.update(val[1]['Precision'], n)
        self.Contingency_Recall.update(val[1]['Recall'], n)
        self.Contingency_F1.update(val[1]['F1_Score'], n)
        self.Contingency_Accuracy.update(val[1]['Accuracy'], n)

        self.Temporal_Precision.update(val[2]['Precision'], n)
        self.Temporal_Recall.update(val[2]['Recall'], n)
        self.Temporal_F1.update(val[2]['F1_Score'], n)
        self.Temporal_Accuracy.update(val[2]['Accuracy'], n)

        self.Expansion_Precision.update(val[3]['Precision'], n)
        self.Expansion_Recall.update(val[3]['Recall'], n)
        self.Expansion_F1.update(val[3]['F1_Score'], n)
        self.Expansion_Accuracy.update(val[3]['Accuracy'], n)

class AverageBinaryMeterPRFA():
    def __init__(self):
        self.Binary_Precision = AverageMeter()
        self.Binary_Recall = AverageMeter()
        self.Binary_F1 = AverageMeter()
        self.Binary_Accuracy = AverageMeter()

        self.Others_Precision = AverageMeter()
        self.Others_Recall = AverageMeter()
        self.Others_F1 = AverageMeter()
        self.Others_Accuracy = AverageMeter()

    def update(self, val, n=1):
        self.Binary_Precision.update(val[0]['Precision'], n)
        self.Binary_Recall.update(val[0]['Recall'], n)
        self.Binary_F1.update(val[0]['F1_Score'], n)
        self.Binary_Accuracy.update(val[0]['Accuracy'], n)

        self.Others_Precision.update(val[1]['Precision'], n)
        self.Others_Recall.update(val[1]['Recall'], n)
        self.Others_F1.update(val[1]['F1_Score'], n)
        self.Others_Accuracy.update(val[1]['Accuracy'], n)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Calculating Precision, Recall, F1 score and Accuracy
def prf_multi_classify(output, target, topk=(1,), classify_num=4):
    result = {}
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.view(1, -1).expand_as(pred)

    for class_num in range(classify_num):
        TP, TN, FN, FP = 0., 0., 0., 0.
        for i in range(batch_size):
            _predict = pred[0][i].data
            _target = target[0][i].data

            # TP    predict 和 target 同时为classify_num
            TP += ((_predict == class_num) & (_target == class_num)).float()
            # TN    predict 和 target 同时不为classify_num
            TN += ((_predict != class_num) & (_target != class_num)).float()
            # FN    predict不是classify_num target是classify_num
            FN += ((_predict != class_num) & (_target == class_num)).float()
            # FP    predict是classify_num target不是classify_num
            FP += ((_predict == class_num) & (_target != class_num)).float()

        p = float('%.4f' % (TP / (TP + FP)).item()) if (TP + FP) != 0 else 0.
        r = float('%.4f' % (TP / (TP + FN)).item()) if (TP + FN) != 0 else 0.
        F1 = float('%.4f' % (2 * r * p / (r + p))) if (r + p) != 0 else 0.
        acc = float('%.4f' % ((TP + TN) / (TP + TN + FP + FN)).item()) if (TP + TN + FP + FN) != 0 else 0.

        result[class_num] = {
            'Precision': p,
            'Recall': r,
            'F1_Score': F1,
            'Accuracy': acc,
        }

    return result


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 8 epochs"""
    lr = lr * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
