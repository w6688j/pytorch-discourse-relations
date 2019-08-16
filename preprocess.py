import argparse

import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='rnn', help='data of model')
args = parser.parse_args()

PDTB_DATA_TRAIN_PATH = 'data/pdtb/train.txt'
PDTB_DATA_TEST_PATH = 'data/pdtb/test.txt'
PROCESSED_DATA_PATH = 'data/processed_data'

if __name__ == '__main__':
    util.create_pdtb_tsv_file(PDTB_DATA_TRAIN_PATH, PROCESSED_DATA_PATH + '/train.tsv')
    util.create_pdtb_tsv_file(PDTB_DATA_TEST_PATH, PROCESSED_DATA_PATH + '/test.tsv')
    util.create_pdtb_tsv_file_type_rnnatt17(PDTB_DATA_TRAIN_PATH, PROCESSED_DATA_PATH, 'train.tsv')
    util.create_pdtb_tsv_file_type_rnnatt17(PDTB_DATA_TEST_PATH, PROCESSED_DATA_PATH, 'test.tsv')
    util.create_pdtb_tsv_file_type_grn16(PDTB_DATA_TRAIN_PATH, PROCESSED_DATA_PATH, 'train.tsv')
    util.create_pdtb_tsv_file_type_grn16(PDTB_DATA_TEST_PATH, PROCESSED_DATA_PATH, 'test.tsv')
    util.create_pdtb_tsv_file_grn16(PDTB_DATA_TRAIN_PATH, PROCESSED_DATA_PATH, 'train.tsv')
    util.create_pdtb_tsv_file_grn16(PDTB_DATA_TEST_PATH, PROCESSED_DATA_PATH, 'test.tsv')
