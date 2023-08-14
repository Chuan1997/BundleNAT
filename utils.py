import random
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from operator import itemgetter
import scipy.sparse as sp
import torch

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_user_vocab():
    user_ids = [line.strip() for line in open('.../user_ids.txt', 'r').read().splitlines()]
    # user_ids = [line.strip() for line in
    #             open('D:/Business/No.3/TransGen/data/ml100k/user_ids.txt', 'r').read().splitlines()]
    user2idx = {int(user): idx for idx, user in enumerate(user_ids)}
    idx2user = {idx: int(user) for idx, user in enumerate(user_ids)}
    return user2idx, idx2user


def load_item_vocab():
    item_ids = [line.strip() for line in open('.../item_ids.txt', 'r').read().splitlines()]
    item2idx = {int(item): idx for idx, item in enumerate(item_ids)}
    idx2item = {idx: int(item) for idx, item in enumerate(item_ids)}
    return item2idx, idx2item


def normalize_np(mx: np.ndarray) -> np.ndarray:
    # 对每一行进行归一化
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    # rows_inv = np.sqrt(rows_inv)
    rows_mat_inv = np.diag(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = rows_mat_inv.dot(mx)  # .dot(cols_mat_inv)
    return mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_gen_data(file_path, matrix,num_item):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    load_dict = np.load(matrix,allow_pickle=True).item()
    USER, CARD, CARD_IDX, ITEM_CAND, Matrix, MASK= [], [], [], [],[],[] #, LABEL, []

    # data_clean = []
    with open(file_path, 'r') as fin:
        for line in fin:
            # data_slice = []
            strs = line.strip().split('\t')
            USER.append(int(strs[0]))
            card_ = [int(x) for x in strs[1].split(',')]
            CARD.append(card_)
            item_cand_ = sorted([int(x) for x in strs[2].split(',')])
            part_matrix = np.array(itemgetter(*item_cand_)(load_dict))
            part_matrix = sp.vstack(part_matrix).tocsr()
            part_matrix = part_matrix.dot(part_matrix.T)
            norm_matrix = normalize(part_matrix).toarray().astype(np.float32)
            ITEM_CAND.append(item_cand_) # sorted
            Matrix.append(norm_matrix)

            indexs = np.array(item_cand_)  # 标签索引
            label = np.zeros((1,num_item), dtype=np.int32)  # 创建具有10个标签的onehot
            label[:,indexs] = 1
            MASK.append(label)

    return np.array(USER), np.array(CARD), np.array(ITEM_CAND), np.array(Matrix), np.array(MASK)

def load_gen_data1(file_path, matrix,num_item):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    load_dict = np.load(matrix,allow_pickle=True).item()
    USER, CARD, CARD_IDX, ITEM_CAND, Matrix, MASK= [], [], [], [],[],[] #, LABEL, []

    # data_clean = []
    with open(file_path, 'r') as fin:
        for line in fin:
            # data_slice = []
            strs = line.strip().split('\t')
            USER.append(int(strs[0]))
            card_ = [int(x) for x in strs[1].split(',')]
            print(card_)
            CARD.append(card_)
            item_cand_ = sorted([int(x) for x in strs[2].split(',')])
            part_matrix = np.array(itemgetter(*item_cand_)(load_dict))
            part_matrix = sp.vstack(part_matrix).tocsr()
            part_matrix = part_matrix.dot(part_matrix.T)
            norm_matrix = normalize(part_matrix).toarray().astype(np.float32)
            ITEM_CAND.append(item_cand_) # sorted
            Matrix.append(norm_matrix)

            indexs = np.array(item_cand_)  # 标签索引
            label = np.zeros((1,num_item), dtype=np.int32)  # 创建具有10个标签的onehot
            label[:,indexs] = 1
            MASK.append(label)

    return np.array(USER), np.array(CARD), np.array(ITEM_CAND), np.array(Matrix), np.array(MASK)

class training_set(Dataset):
    def __init__(self, USER, CARD, ITEM_CAND, Matrix, MASK): #, LABEL, CARD_IDX
        self.USER = USER                        # set data
        self.CARD = CARD
        # self.CARD_IDX = CARD_IDX
        self.ITEM_CAND = ITEM_CAND
        self.Matrix = Matrix # set lables
        self.MASK = MASK
    def __len__(self):
        return len(self.USER)                   # return length

    def __getitem__(self, idx):
        return [self.USER[idx], self.CARD[idx], self.ITEM_CAND[idx],self.Matrix[idx],self.MASK[idx]]

def load_gen_data_NCF(file_path,num_item):
    user2idx, _ = load_user_vocab()
    item2idx, _ = load_item_vocab()

    USER, CARD, CARD_IDX, ITEM_CAND, MASK, USERl= [], [], [], [],[],[]

    # data_clean = []
    with open(file_path, 'r') as fin:
        for line in fin:
            strs = line.strip().split('\t')
            USER.append(int(strs[0]))
            USERl.append([int(strs[0])]*100)
            # USERl.append([int(strs[0])] * 200)
            card_ = [int(x) for x in strs[1].split(',')]
            CARD.append(card_)
            item_cand_ = sorted([int(x) for x in strs[2].split(',')])
            ITEM_CAND.append(item_cand_) # sorted
            # Matrix.append(norm_matrix)

            indexs = np.array(item_cand_)  # 标签索引
            label = np.zeros((1,num_item), dtype=np.int32)  # 创建具有10个标签的onehot
            label[:,indexs] = 1
            MASK.append(label)

    return np.array(USER), np.array(CARD), np.array(ITEM_CAND), np.array(MASK), np.array(USERl) #, np.array(Matrix)



class training_set_NCF(Dataset):
    def __init__(self, USER, CARD, ITEM_CAND, MASK,  USERl):#Matrix,
        self.USER = USER
        self.CARD = CARD
        self.ITEM_CAND = ITEM_CAND
        #self.Matrix = Matrix
        self.MASK = MASK
        self.USERl = USERl

    def __len__(self):
        return len(self.USER)

    def __getitem__(self, idx):
        return [self.USER[idx], self.CARD[idx], self.ITEM_CAND[idx],self.MASK[idx],self.USERl[idx]] #,self.Matrix[idx]

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

import torch
def featureread(path1, path2):
    t1 = np.load(path1, allow_pickle=True)
    t2 = np.load(path2, allow_pickle=True)
    t_emb2 = torch.from_numpy(np.array(t1))
    d_emb2 = torch.from_numpy(np.array(t2))
    return t_emb2, d_emb2#,it
