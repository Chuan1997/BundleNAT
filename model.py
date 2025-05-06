import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import log
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn import Parameter
from layers import GraphConvolution
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        #print(nhid)
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def _mask(self):
        return self.mask

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def l2_loss(input):
    return torch.sum(input ** 2)/2

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

def create_matrix(sim,n):
    matrix = torch.cosine_similarity(sim.unsqueeze(2), sim.unsqueeze(1), dim=-1).cpu()
    mask = np.argpartition(matrix, -n, axis=1) < matrix.shape[1] - n
    matrix[mask] = 0

    mat = F.normalize(matrix.view(matrix.size(0), matrix.size(1) * matrix.size(2)), p=1, dim=1)
    mat = mat.view(*matrix.size())

    return mat.cuda()

#多头注意力层
class MultiHeadAttentionLayer( nn.Module ):

    def __init__(self, e_dim, h_dim, n_heads):
        '''
        :param e_dim: 输入的向量维度
        :param h_dim: 每个单头注意力层输出的向量维度
        :param n_heads: 头数
        '''
        super().__init__()
        self.atte_layers = nn.ModuleList( [ OneHeadAttention( e_dim, h_dim ) for _ in range( n_heads ) ] )
        self.l = nn.Linear( h_dim * n_heads, e_dim)

    def forward(self, seq_inputs, querys=None, mask=None):
        outs = []
        for one in self.atte_layers:
            out = one(seq_inputs, querys, mask)
            outs.append(out)
        # [ batch, seq_lens, h_dim * n_heads ]
        outs = torch.cat(outs, dim=-1)
        # [ batch, seq_lens, e_dim ]aa,
        outs = self.l(outs)
        return outs

class MultiHeadAttentionLayer_i( nn.Module ):
    def __init__(self, e_dim, h_dim, n_heads):
        super().__init__()
        self.atte_layers = nn.ModuleList( [ OneHeadAttention_i( e_dim, h_dim ) for _ in range( n_heads ) ] )
        self.l = nn.Linear( h_dim * n_heads, e_dim)

    def forward(self, seq_inputs, querys=None, mask=None):
        outs = []
        for one in self.atte_layers:
            out = one(seq_inputs,querys, mask)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        outs = self.l(outs)
        return outs

class MultiHeadAttentionLayer_dec( nn.Module ):

    def __init__(self, e_dim, h_dim, n_heads):
        '''
        :param e_dim: 输入的向量维度
        :param h_dim: 每个单头注意力层输出的向量维度
        :param n_heads: 头数
        '''
        super().__init__()
        self.atte_layers = nn.ModuleList( [ OneHeadAttention_dec( e_dim, h_dim ) for _ in range( n_heads ) ] )
        self.l = nn.Linear( h_dim * n_heads, e_dim)

    def forward(self, seq_inputs, querys=None, mask=None):
        outs = []
        for one in self.atte_layers:
            out = one(seq_inputs, querys, mask)
            outs.append(out)
        # [ batch, seq_lens, h_dim * n_heads ]
        outs = torch.cat(outs, dim=-1)
        # [ batch, seq_lens, e_dim ]
        outs = self.l(outs)
        return outs

#单头注意力层
class OneHeadAttention( nn.Module ):

    def __init__( self, e_dim, h_dim ):
        '''
        :param e_dim: 输入向量维度
        :param h_dim: 输出向量维度
        '''
        super().__init__()
        self.h_dim = h_dim
        self.length = 100
        self.kt = 10
        # 初始化Q,K,V的映射线性层
        self.lQ = nn.Linear( e_dim, h_dim )
        self.lK = nn.Linear( e_dim, h_dim )
        self.lV = nn.Linear( e_dim, h_dim )

    def forward(self, seq_inputs, querys=None, mask=None):
        '''
        :param seq_inputs: #[ batch, seq_lens, e_dim ]
        :param querys: #[ batch, seq_lens, e_dim ]
        :param mask: #[ 1, seq_lens, seq_lens ] or [ 1, 1, seq_lens ]
        :return:
        '''
        # 如果有encoder的输出, 则映射该张量，否则还是就是自注意力的逻辑
        if querys is not None:
            Q = self.lQ( querys ) #[ batch, seq_lens, h_dim ]
        else:
            Q =  self.lQ( seq_inputs ) #[ batch, seq_lens, h_dim ]
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( seq_inputs ) #[ batch, seq_lens, h_dim ]
        # [ batch, seq_lens, seq_lens ]
        QK = torch.matmul(Q, K.permute(0, 2, 1))
        # [ batch, seq_lens, seq_lens ]
        QK /= ( self.h_dim ** 0.5 )
        # QK= QK + graph
        # 将对应Mask序列中0的位置变为-1e9,意为遮盖掉此处的值
        if mask is not None:
            QK = QK.masked_fill( mask == 0, -1e9 )
        # [ batch, seq_lens, seq_lens ]
        a = torch.softmax( QK, dim = -1 )
        # [ batch, seq_lens, h_dim ]a,
        outs = torch.matmul( a, V )
        return outs

class OneHeadAttention_i( nn.Module ):

    def __init__( self, e_dim, h_dim ):
        super().__init__()
        self.h_dim = h_dim

        # 初始化Q,K,V的映射线性层
        self.lQ = nn.Linear( e_dim, h_dim )
        self.lK = nn.Linear( e_dim, h_dim )
        self.lV = nn.Linear( e_dim, h_dim )

    def forward( self, seq_inputs , querys = None, mask = None ):
        if querys is not None:
            Q = self.lQ( querys ) #[ batch, seq_lens, h_dim ]
        else:
            Q =  self.lQ( seq_inputs ) #[ batch, seq_lens, h_dim ]
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( seq_inputs ) #[ batch, seq_lens, h_dim ]
        QK = torch.matmul( Q,K.permute( 0, 2, 1 ) )
        QK /= ( self.h_dim ** 0.5 )

        if mask is not None:
            QK = QK.masked_fill( mask == 0, -1e9 )
        a = torch.softmax( QK, dim = -1 )
        outs = torch.matmul( a, V )
        return outs

class OneHeadAttention_dec( nn.Module ):

    def __init__( self, e_dim, h_dim ):
        '''
        :param e_dim: 输入向量维度
        :param h_dim: 输出向量维度
        '''
        super().__init__()
        self.h_dim = h_dim

        # 初始化Q,K,V的映射线性层
        self.lQ = nn.Linear( e_dim, h_dim )
        self.lK = nn.Linear( e_dim, h_dim )
        self.lV = nn.Linear( e_dim, h_dim )

    def forward( self, seq_inputs , querys = None, mask = None ):
        '''
        :param seq_inputs: #[ batch, seq_lens, e_dim ]
        :param querys: #[ batch, seq_lens, e_dim ]
        :param mask: #[ 1, seq_lens, seq_lens ] or [ 1, 1, seq_lens ]
        :return:
        '''
        # 如果有encoder的输出, 则映射该张量，否则还是就是自注意力的逻辑
        if querys is not None:
            Q = self.lQ( querys ) #[ batch, seq_lens, h_dim ]
        else:
            Q =  self.lQ( seq_inputs ) #[ batch, seq_lens, h_dim ]
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( querys ) #[ batch, seq_lens, h_dim ]
        # [ batch, seq_lens, seq_lens ]
        QK = torch.matmul( K, Q.permute( 0, 2, 1 ))
        # [ batch, seq_lens, seq_lens ]
        QK /= ( self.h_dim ** 0.5 )

        # 将对应Mask序列中0的位置变为-1e9,意为遮盖掉此处的值
        if mask is not None:
            QK = QK.masked_fill( mask == 0, -1e9 )
        # [ batch, seq_lens, seq_lens ]
        a = torch.softmax( QK, dim = -1 )
        # [ batch, seq_lens, h_dim ]
        outs = torch.matmul( a, V )
        return outs


#前馈神经网络
class FeedForward(nn.Module):

    def __init__( self, e_dim, ff_dim, drop_rate = 0.1 ):
        super( ).__init__( )
        self.l1 = nn.Linear( e_dim, ff_dim )
        self.l2 = nn.Linear( ff_dim, e_dim )
        self.drop_out = nn.Dropout( drop_rate )

    def forward( self, x ):
        outs = self.l1( x )
        outs = self.l2( self.drop_out( torch.relu( outs ) ) )
        return outs


#编码层
class EncoderLayer(nn.Module):

    def __init__( self, e_dim, h_dim, n_heads, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()
        # 初始化多头注意力层
        self.attention = MultiHeadAttentionLayer( e_dim, h_dim, n_heads )
        # 初始化注意力层之后的LN
        self.a_LN = nn.LayerNorm( e_dim )
        # 初始化前馈神经网络层
        self.ff_layer = FeedForward( e_dim, e_dim//2 )
        # 初始化前馈网络之后的LN
        self.ff_LN = nn.LayerNorm( e_dim )

        self.drop_out = nn.Dropout( drop_rate )

    def forward(self, seq_inputs):
        outs_ = self.attention(seq_inputs)
        # 残差连与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.a_LN( seq_inputs + self.drop_out( outs_ ) )
        # 前馈神经网络, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.ff_layer( outs )
        # 残差与LN, 输出维度[ batch, seq_lens, e_dim ]aa,
        outs = self.ff_LN( outs + self.drop_out( outs_) )
        return outs


class TransformerEncoder( nn.Module ):

    def __init__(self, e_dim, h_dim, n_heads, n_layers, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param n_layers: 编码层的数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()

        #初始化N个“编码层”
        self.encoder_layers = nn.ModuleList( [EncoderLayer( e_dim, h_dim, n_heads, drop_rate )
                                         for _ in range( n_layers )] )
        
    def forward(self, seq_inputs):
        '''
        :param seq_inputs: 已经经过Embedding层的张量，维度是[ batch, seq_lens, dim ]
        :return: 与输入张量维度一样的张量，维度是[ batch, seq_lens, dim ]
        '''
        # aa=[]
        #输入进N个“编码层”中开始传播
        for layer in self.encoder_layers:
            seq_inputs = layer(seq_inputs)
          

        return seq_inputs

def get_normalized_probs(net_output):     #"""Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output.float()
        return F.log_softmax(logits, dim=-1)

#生成mask序列
def _mask_user( size ):
    targets=torch.ones(1,size,size)
    inx = torch.LongTensor([[0] * size]).reshape(1, -1, 1)
    targets.scatter_(2,inx,0)
    return targets

#解码层
class DecoderLayer(nn.Module):

    def __init__( self, e_dim, h_dim, n_heads, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param querys: encoder的输出
        :param drop_rate: drop out的比例
        '''
        super().__init__()

        # 初始化自注意力层
        self.self_attention = MultiHeadAttentionLayer_i( e_dim, h_dim, n_heads )
        # 初始化自注意力层之后的LN
        self.sa_LN = nn.LayerNorm( e_dim )
        # 初始化交互注意力层
        self.interactive_attention = MultiHeadAttentionLayer_dec( e_dim, h_dim, n_heads )
        # 初始化交互注意力层之后的LN
        self.ia_LN = nn.LayerNorm (e_dim )
        # 初始化前馈神经网络层
        self.ff_layer = FeedForward( e_dim, e_dim//2 )
        # 初始化前馈网络之后的LN
        self.ff_LN = nn.LayerNorm( e_dim )

        self.drop_out = nn.Dropout( drop_rate )

    def forward( self, seq_inputs ,querys): #, mask
        '''
        :param seq_inputs: [ batch, seqs_len, e_dim ]
        :param mask: 遮盖位置的标注序列 [ 1, seqs_len, seqs_len ]
        '''
        # 自注意力层, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.self_attention(seq_inputs, mask= None)
        # 残差连与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.sa_LN( seq_inputs + self.drop_out( outs_ ) )
        # print(outs.size())
        # 交互注意力层, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.interactive_attention( outs, querys )
        # print(outs_.size())
        # 残差连与LN, 输出维度[ batch, seq_lens, e_dim
        outs = self.ia_LN( outs + self.drop_out(outs_) )
        # 前馈神经网络, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.ff_layer( outs )
        # 残差与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.ff_LN( outs + self.drop_out( outs_) )
        return outs



class TransformerDecoder(nn.Module):

    def __init__(self, e_dim, h_dim, n_heads, n_layers, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param n_layers: 解码层的数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()

        # 初始化N个“解码层”
        self.decoder_layers = nn.ModuleList( [DecoderLayer( e_dim, h_dim, n_heads, drop_rate )
                                         for _ in range( n_layers )] )
    def forward( self, seq_inputs, querys): #, mask, pe
        '''
        :param seq_inputs: 已经经过Embedding层的张量，维度是[ batch, seq_lens, dim ]
        :return: 与输入张量维度一样的张量，维度是[ batch, seq_lens, dim ]
        '''
        # 先进行位置编码
        # seq_inputs = self.position_encoding( seq_inputs )

        # 得到mask序列
        # mask = subsequent_mask( seq_inputs.shape[1] )
        # seq_inputs = seq_inputs + pe
        # 输入进N个“解码层”中开始传播
        for layer in self.decoder_layers:
            # seq_inputs = layer( seq_inputs, querys, mask )
            seq_inputs = layer(seq_inputs, querys)

        return seq_inputs



class FeedForward_Gen(nn.Module):

    def __init__(self, e_dim, ff_dim, drop_rate=0.1):#, p_dim
        super().__init__()
        self.l1 = nn.Linear(e_dim, ff_dim)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.flatten(1)
        outs = self.l1(x)
        outs = self.drop_out(torch.relu(outs))
        return outs

class FeatureFusionGate(nn.Module):
    def __init__(self, embedding_dim):
        super(FeatureFusionGate, self).__init__()
        self.linear1 = nn.Linear(embedding_dim*4, embedding_dim*2)
        self.linear2 = nn.Linear(embedding_dim*4, embedding_dim*2)

    def forward(self, colla, cooc):
        concat_feature = torch.cat((colla, cooc), dim=2)
        fusion = F.elu(self.linear1(concat_feature))
        gate = F.sigmoid(self.linear2(concat_feature))
        out = torch.mul(gate, fusion) + torch.mul((1-gate), cooc)
        # print(gate)
        return out

class Transformer4Gen( nn.Module ):

    def __init__( self, n_users, n_items, u_fea, i_fea, num_tags, all_seq_lens, full_length, e_dim = 64, n_heads = 4, n_layers = 2 ,alpha = 0.2):
        '''
        :param n_items: 总物品数量
        :param all_seq_lens: 序列总长度，包含历史物品序列及目标物品
        :param e_dim: 向量维度
        :param n_heads: Transformer中多头注意力层的头目数
        :param n_layers: Transformer中的encoder_layer层数
        :param alpha: 辅助损失函数的计算权重
        '''
        super(Transformer4Gen, self).__init__()
        self.users = nn.Embedding( n_users, e_dim)
        self.items = nn.Embedding( n_items, e_dim)

        self.items_GNN = nn.Embedding(n_items, e_dim*2, max_norm=1)
        self.users.weight.data.copy_(u_fea)
        self.items.weight.data.copy_(i_fea)
        self.users.weight.requires_grad = True
        self.items.weight.requires_grad = True

        self.trigger = nn.Embedding(1, e_dim*2, max_norm=1)
        self.featureFusionGate = FeatureFusionGate(e_dim)

        self.encoder = TransformerEncoder( e_dim*2, e_dim, n_heads, n_layers )

        self.length = all_seq_lens
        self.tleng = num_tags

        self.predict_layer = FeedForward_Gen(e_dim * 2, full_length)
        self.dropout = 0.1
        self.pool = nn.AvgPool2d(kernel_size=(self.length,1))
        # self.pool = nn.MaxPool2d(kernel_size=(self.length, 1))
        self.GCN = GCN(e_dim*2, e_dim*2, e_dim*2, self.dropout)

        # Maps the output of the Decoder into tag space.
        self.hidden2tag = nn.Sequential(nn.Linear(e_dim, num_tags), nn.LogSoftmax(dim=2)) #*all_seq_lens

        self.decoder = TransformerDecoder(e_dim*2, e_dim, n_heads, n_layers) #e_dim*2, e_dim

        self.alpha = alpha
        self.padding_idx = -1
        self.n_items = n_items

    def compute_loss(self, net_output, target):
        # Bipart Loss
        bs, seq_len = target.size()
        target = target.repeat(1, seq_len).view(bs, seq_len, seq_len)
        _, prelen = net_output.squeeze(1).size()
        net_output = net_output.squeeze(1).repeat(1, seq_len).view(bs, seq_len, prelen)
        bipart_no_pad = target.ne(self.padding_idx)
        bipart_lprobs = get_normalized_probs(net_output)

        nll_loss = -bipart_lprobs.gather(dim=-1, index=target)  # bs seq seq
        nll_loss = nll_loss * bipart_no_pad

        best_match = np.repeat(np.arange(seq_len).reshape(1, -1, 1), bs, axis=0)  # np.zeros((bs, seq_len, 1))
        nll_loss_numpy = nll_loss.detach().cpu().numpy() #.cpu()

        for batch_id in range(bs):
            no_pad_num = bipart_no_pad[batch_id, 0].sum()
            raw_index, col_index = lsa(nll_loss_numpy[batch_id, :no_pad_num, :no_pad_num])
            best_match[batch_id, :no_pad_num] = col_index.reshape(-1, 1)

        best_match = torch.Tensor(best_match).to(target).long()
        nll_loss = nll_loss.gather(dim=-1, index=best_match)
        nll_loss = nll_loss.squeeze(-1)
        return nll_loss.sum()

    def forward( self, x, u , target_idx, length, adj,mask):
        # [ batch_size, seqs_len, dim ]
        item_embs = self.items(x)
        item_sim = self.items_GNN(x)
        posi = self.GCN(item_sim,adj)
        user_embs = self.users(u)
        uu = torch.stack(self.length * [user_embs], axis=2).squeeze() # print(uu.size()) # print(item_embs.size())
        seq_embs = torch.cat([uu, item_embs], axis=2) # print(seq_embs.size())
        seq_embs_f = seq_embs+posi
        enc_embs = self.encoder(seq_embs_f)

        trigger = self.pool(enc_embs)

        outs = self.decoder(trigger, enc_embs)

        logit = self.predict_layer(outs)
        recloss = self.compute_loss(logit, target_idx)
        return recloss

    def inference(self,x, u, target_idx, length, adj, mask):#
        item_embs = self.items(x)
        user_embs = self.users(u)
        uu = torch.stack(self.length * [user_embs], axis=2).squeeze()
        seq_embs = torch.cat([uu, item_embs], axis=2)
        item_sim = self.items_GNN(x)
        posi = self.GCN(item_sim, adj)
        seq_embs_f = seq_embs + posi
        enc_embs = self.encoder(seq_embs_f)

        trigger = self.pool(enc_embs)
        outs = self.decoder(trigger, enc_embs)
        logit = self.predict_layer(outs)
        _, list_pos = torch.topk(logit, k=20) # 5

        return enc_embs, list_pos




