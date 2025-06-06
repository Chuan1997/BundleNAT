import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Transformer4Gen,_mask_user
from utils1 import load_gen_data1, training_set, featureread
import math
from torch import nn
import numpy as np
import time
import os

def to_array(*args, **kwargs):
    a = []
    for v in args:
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, **kwargs)
        a.append(v)
    return tuple(a)

def to_tensor(*args, **kwargs):
    a = []
    for v in args:
        if not isinstance(v, torch.Tensor):
            v = torch.Tensor(v, **kwargs)
        a.append(v)
    return tuple(a)

def bundle_precision(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    precision = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        precision += float(y_pred[i][0] in a)
        common.append(len(a))
    precision /= len(y_true)
    return precision


def precision_at_4(card_infer, item_pos):
    score = []
    for i in range(len(card_infer)):
        res = 0.0
        for item in item_pos[i]:
            if item in card_infer[i]:
                res += 1.0
        score.append(res / (1.0 * len(card_infer[0])))
    return score

def bundle_precision_plus(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    precision_plus = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        precision_plus += float(len(a) > 0)
        common.append(len(a))
    precision_plus /= len(y_true)
    return precision_plus


def bundle_recall(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    recall = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        recall += len(a) / len(y_true[0])
        common.append(len(a))
    recall /= len(y_true)
    return recall


def doEva_train(net, dataloader, batchSize):
    precision_score = []
    p1_score = []
    p2_score = []
    r1_score = []
    for data in tqdm(DataLoader(dataloader, batch_size = batchSize, shuffle = True)):
        u = data[0].unsqueeze(1).long().cuda()
        cand = data[1].long().cuda()
        cpool = data[2].long().cuda()
        length = torch.LongTensor([0,1,2,3,4]).cuda()
        ADJ = data[3].cuda()
        mask = data[4].cuda()
        with torch.no_grad():
            _, out = net.inference(cpool, u, cand, length,ADJ,mask)
            out1 = out.squeeze(1).detach().cpu().numpy().tolist()
            cand1 = cand.detach().cpu().numpy().tolist()
            precision_s = precision_at_4(out1, cand1)
            p1 = bundle_precision(cand1, out1)
            p2= bundle_precision_plus(cand1, out1)
            r1 =bundle_recall(cand1, out1)

            precision_score.extend(precision_s)
            p1_score.append(p1)
            p2_score.append(p2)
            r1_score.append(r1)

    return sum(precision_score)/len(precision_score), sum(p1_score)/len(p1_score), sum(p2_score)/len(p2_score), sum(r1_score)/len(r1_score)

def doEva_test(net, dataloader, batchSize,epoch):
    net.eval()
    precision_score = []
    p1_score = []
    p2_score = []
    r1_score = []
    cc=0
    for data in tqdm(DataLoader(dataloader, batch_size = batchSize, shuffle = False)):
        u = data[0].unsqueeze(1).long().cuda()
        cand = data[1].long().cuda()
        cpool = data[2].long().cuda()
        length = torch.LongTensor([0,1,2,3,4]).cuda()

        ADJ = data[3].cuda()
        mask = data[4].cuda()

        with torch.no_grad():
            start=time.time()
            embs,out = net.inference(cpool, u, cand, length,ADJ,mask)
            end=time.time()
            print((end-start)/batchSize)

            out1 = out.squeeze(1).detach().cpu().numpy().tolist()
            cand1 = cand.detach().cpu().numpy().tolist()
            precision_s = precision_at_4(out1, cand1)

            precision_score.extend(precision_s)
            p1 = bundle_precision(cand1, out1)
            p2 = bundle_precision_plus(cand1, out1)
            r1 = bundle_recall(cand1, out1)

            precision_score.extend(precision_s)
            p1_score.append(p1)
            p2_score.append(p2)
            r1_score.append(r1)
            cc+=1

    return sum(precision_score) / len(precision_score), sum(p1_score) / len(p1_score), sum(p2_score) / len(p2_score), sum(
            r1_score) / len(r1_score)


def train(train, test, testsize, n_user, n_item, u_fea, i_fea,num_tag, all_seq_lens, full_length, epochs = 50, batchSize = 256, lr = 0.001, dim = 128, eva_per_epochs = 2):
    #初始化模型
    net = Transformer4Gen(n_user, n_item, u_fea, i_fea, num_tag, all_seq_lens,full_length)
    net.cuda()
    #初始化优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    norm = nn.Softmax(dim=1)
    best_re=0.0
    #开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in tqdm(DataLoader(train, batch_size = batchSize, shuffle = True)):
            u = seq[0].unsqueeze(1).long().cuda()# print(u.size())
            card = seq[1].long().cuda()
            length = torch.LongTensor([0,1,2,3,4]).cuda()
            item_cand = seq[2].long().cuda()
            ADJ=seq[3].cuda()
            mask=seq[4].cuda()

            optimizer.zero_grad()
            loss = net(item_cand, u, card, length,ADJ,mask)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e,all_lose/(len(train)//batchSize)))
        loss_log.write('{}\n'.format(
            all_lose/(len(train)//batchSize)))
        loss_log.flush()

        #评估模型 调整一下
        if e % eva_per_epochs == 0 or e == 49:
            p0, p1, p2, r1 = doEva_train(net, train, batchSize)
            print('train:precision-s:{:.4f}, precision-1:{:.4f}, precision-2:{:.4f}, recall-s:{:.4f}'.format(p0,p1,p2,r1))
            train_log.write('{}\t{}\t{}\n'.format(
                p1,p2,r1))
            train_log.flush()
            t0, t1, t2, tt1 = doEva_test(net, test, testsize,e)
            if tt1>best_re:
                best_re=tt1
            print('test:precision-s:{:.4f}, precision-1:{:.4f}, precision-2:{:.4f}, recall-s:{:.4f}'.format(t0,t1,t2,tt1))
            test_log.write('{}\t{}\t{}\n'.format(
                t1,t2,tt1))
            test_log.flush()


if __name__ == '__main__':
    torch.cuda.manual_seed_all(1024)
    testsize = 256  # len(USER_t)
    all_seq_lens = 200 # 候选物品集合长度
    full_length = 32770 # 物品集合大小
    num_tag = 20 # 目标捆绑包大小
    num_user = 8006 #
    num_item = 32770 #
    t1 = time.time()
    USER, CARD, ITEM_CAND, MATRIX , MASK= load_gen_data1('.../rerank_data_train.txt','.../matrix.npy',num_item)
    USER_t, CARD_t, ITEM_CAND_t, MATRIX_t, MASK_t = load_gen_data1('.../rerank_data_test.txt','.../matrix.npy',num_item )

    t2 = time.time() - t1
    print(t2)

    training_dataset = training_set(USER, CARD, ITEM_CAND,MATRIX, MASK)
    testing_dataset = training_set(USER_t, CARD_t, ITEM_CAND_t,MATRIX_t, MASK_t)

    u_e, i_e = featureread('.../ue.npy','.../ie.npy')
    logdir='.../log/'
    gen_train_log_path = 'gen_train_log.txt'
    gen_test_log_path = 'gen_test_log.txt'
    gen_loss_log_path='loss.txt'

    train_log = open(os.path.join(logdir, gen_train_log_path), 'w')
    train_log.write('precision\tprecision+\trecall\n')
    test_log = open(os.path.join(logdir, gen_test_log_path), 'w')
    test_log.write('precision\tprecision+\trecall\n')

    loss_log = open(os.path.join(logdir, gen_loss_log_path), 'w')
    loss_log.write('loss\n')

    t1=time.time()
    train(training_dataset, testing_dataset, testsize, num_user, num_item, u_e, i_e, num_tag, all_seq_lens,full_length)
    t2 = time.time()-t1
    print(t2)
