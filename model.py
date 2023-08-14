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

# class GCN(nn.Module):
#     def __init__(self, nfeat, out, dropout):
#         #print(nhid)
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, out)
#         self.dropout = dropout
#
#     def _mask(self):
#         return self.mask
#
#     def forward(self, x, adj):
#         x = self.gc1(x, adj)
#         x = F.dropout(x, self.dropout, training=self.training)
#         return x

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
    # matrix = cosine_similarity(sim.cpu(), sim.cpu())
    mask = np.argpartition(matrix, -n, axis=1) < matrix.shape[1] - n
    matrix[mask] = 0
    # matrix[~mask]=1
    # print(matrix)
    mat = F.normalize(matrix.view(matrix.size(0), matrix.size(1) * matrix.size(2)), p=1, dim=1)
    mat = mat.view(*matrix.size())
    # print(mat.shape)
    # mat=normalize(matrix)
    # fin_mat=sparse_mx_to_torch_sparse_tensor(mat)
    # fin_mat=sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(mat.numpy()))
    # print(fin_mat.shape)
    return mat.cuda()

#多头注意力层        # aa=[]            # aa.append(a)
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

    # def forward(self, seq_inputs, GNN_input,graph, querys=None, mask=None):
    # def forward(self, seq_inputs, Q,K, querys=None, mask=None):
    def forward(self, seq_inputs, querys=None, mask=None):
        outs = []
        for one in self.atte_layers:
            # out = one(seq_inputs, GNN_input, graph, querys, mask)
            # out = one(seq_inputs, Q,K, querys, mask)a,
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

    # def forward( self, seq_inputs, GNN_input, graph, querys = None, mask = None ):
    # def forward(self, seq_inputs, Q,K,  querys=None, mask=None):
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
        # print(GNN_input.shape)
        # print(graph.shape)
        # Q=self.GCN(GNN_input,graph)
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( seq_inputs ) #[ batch, seq_lens, h_dim ]
        # [ batch, seq_lens, seq_lens ]
        QK = torch.matmul(Q, K.permute(0, 2, 1))
        # QK *= math.log(self.length, self.kt)
        # QK = torch.matmul( Q,K.permute( 0, 2, 1 ) )
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

    # def forward(self, seq_inputs,GNN_input,graph):
    # def forward(self, seq_inputs,Q,K):
    def forward(self, seq_inputs):
        # seq_inputs = [batch, seqs_len, e_dim]
        # 多头注意力, 输出维度[ batch, seq_lens, e_dim ]
        # outs_ = self.attention( seq_inputs,GNN_input,graph)
        # outs_ = self.attention(seq_inputs, Q,K)aa,
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
    # def forward( self, seq_inputs,GNN_input,graph):
    # def forward(self, seq_inputs, Q,K):
    def forward(self, seq_inputs):
        '''
        :param seq_inputs: 已经经过Embedding层的张量，维度是[ batch, seq_lens, dim ]
        :return: 与输入张量维度一样的张量，维度是[ batch, seq_lens, dim ]
        '''
        # aa=[]
        #输入进N个“编码层”中开始传播
        for layer in self.encoder_layers:
            # seq_inputs = layer(seq_inputs,GNN_input,graph)
            # seq_inputs = layer(seq_inputs, Q,K)a,
            seq_inputs = layer(seq_inputs)
            # aa.append(a)aa,

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
        # outs_ = self.self_attention( seq_inputs, mask = mask)
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

# 前馈神经网络
# class FeedForward_Gen(nn.Module):
#
#     def __init__(self, e_dim, ff_dim, p_dim, drop_rate=0.1):
#         super().__init__()
#         self.l1 = nn.Linear(e_dim, ff_dim)
#         self.l2 = nn.Linear(ff_dim, p_dim)
#         self.drop_out = nn.Dropout(drop_rate)
#
#     def forward(self, x):
#         outs = self.l1(x)
#         outs = self.l2(self.drop_out(torch.relu(outs)))
#         return outs

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
        # self.users = nn.Embedding(n_users, e_dim, max_norm=1)
        # self.items = nn.Embedding(n_items, e_dim, max_norm=1)
        self.items_GNN = nn.Embedding(n_items, e_dim*2, max_norm=1)
        self.users.weight.data.copy_(u_fea)
        self.items.weight.data.copy_(i_fea)
        self.users.weight.requires_grad = True
        self.items.weight.requires_grad = True
        # self.trigger = nn.Embedding(num_tags, e_dim*2, max_norm = 1 )
        # self.trigger = nn.Embedding(num_tags, e_dim, max_norm = 1 ) #调整
        self.trigger = nn.Embedding(1, e_dim*2, max_norm=1)
        self.featureFusionGate = FeatureFusionGate(e_dim)

        self.encoder = TransformerEncoder( e_dim*2, e_dim, n_heads, n_layers )
        # self.encoder = TransformerEncoder(e_dim, e_dim//2, n_heads, n_layers)
        self.length = all_seq_lens
        self.tleng = num_tags
        # self.predict_layer = FeedForward_Gen(e_dim*2, e_dim, full_length)
        # self.predict_layer = FeedForward_Gen(e_dim * self.length * 2, full_length)
        self.predict_layer = FeedForward_Gen(e_dim * 2, full_length)
        self.dropout = 0.1
        self.pool = nn.AvgPool2d(kernel_size=(self.length,1))
        # self.pool = nn.MaxPool2d(kernel_size=(self.length, 1))
        self.GCN = GCN(e_dim*2, e_dim*2, e_dim*2, self.dropout)
        # self.GCN = GCN(e_dim * 2, e_dim * 2, self.dropout)
        # self.GCN1 = GCN(e_dim * 2, e_dim * 2, self.dropout)
        # self.nei=6
        # self.predict_layer = FeedForward_Gen(e_dim, e_dim//2, all_seq_lens)
        # self.triid= torch.LongTensor([0, 1, 2, 3])

        # self.crf = CRF(num_tags, batch_first=True)
        # Maps the output of the Decoder into tag space.
        self.hidden2tag = nn.Sequential(nn.Linear(e_dim, num_tags), nn.LogSoftmax(dim=2)) #*all_seq_lens

        self.decoder = TransformerDecoder(e_dim*2, e_dim, n_heads, n_layers) #e_dim*2, e_dim
        # self.decoder = TransformerDecoder(e_dim, e_dim//2, n_heads, n_layers)
        # self.activation=nn.GELU()
        # self.exter_attn_layer = nn.Sequential(
        #     nn.Linear(self.length, self.length),
        #     self.activation,
        #     nn.Linear(self.length, self.length)
        # )

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
        # Q = self.GCN(item_sim, adj)
        # K = self.GCN1(item_sim, adj)
        user_embs = self.users(u)
        uu = torch.stack(self.length * [user_embs], axis=2).squeeze() # print(uu.size()) # print(item_embs.size())
        seq_embs = torch.cat([uu, item_embs], axis=2) # print(seq_embs.size())
        seq_embs_f = seq_embs+posi
        # seq_embs_f = self.featureFusionGate(seq_embs, posi)am,
        enc_embs = self.encoder(seq_embs_f)

        # enc_embs = self.encoder(posi)
        # adj = self.exter_attn_layer(adj)
        # enc_embs = self.encoder(seq_embs,Q,Q) # enc_embs.size()
        # enc_embs = self.encoder(seq_embs)
        # trigger_f = self.trigger(length)
        trigger = self.pool(enc_embs)
        # trigger = self.trigger(length)
        # trigger_u = torch.stack(self.tleng * [user_embs], axis=2).squeeze()
        # trigger = torch.stack([self.trigger(u)]*self.tleng, axis=2).squeeze()

        outs = self.decoder(trigger, enc_embs)
        # outs = self.decoder(posi, enc_embs)
        # out = outs.reshape(outs.size(0),1,-1)

        # logit = self.predict_layer(out)
        logit = self.predict_layer(outs)
        recloss = self.compute_loss(logit, target_idx)
        return recloss

    def inference(self,x, u, target_idx, length, adj, mask):#, mask
        item_embs = self.items(x)
        user_embs = self.users(u)
        uu = torch.stack(self.length * [user_embs], axis=2).squeeze()
        seq_embs = torch.cat([uu, item_embs], axis=2)
        item_sim = self.items_GNN(x)
        posi = self.GCN(item_sim, adj)
        # # # Q = self.GCN(item_sim, adj)
        # # # K = self.GCN1(item_sim, adj)
        seq_embs_f = seq_embs + posi
        # seq_embs_f = self.featureFusionGate(seq_embs, posi)am,
        enc_embs = self.encoder(seq_embs_f)
        # enc_embs = self.encoder(seq_embs + posi)
        # adj = self.exter_attn_layer(adj)
        # enc_embs = self.encoder(seq_embs,Q,Q)
        # enc_embs = self.encoder(posi)

        # triid = length.expand_as(target_idx)
        # trigger = self.trigger(triid)
        # trigger_f = self.trigger(length)
        trigger = self.pool(enc_embs)
        # trigger = self.trigger(length)
        # trigger_u = torch.stack(self.tleng * [user_embs], axis=2).squeeze()
        # trigger = torch.stack(self.tleng * [user_embs], axis=2).squeeze()
        # trigger = torch.stack([self.trigger(u)] * self.tleng, axis=2).squeeze()
        # outs = self.decoder(seq_embs, enc_embs, _mask_user(seq_embs.size()[1]))
        outs = self.decoder(trigger, enc_embs)
        # outs = self.decoder(posi, enc_embs)
        # out = outs.reshape(outs.size(0), 1, -1)
        logit = self.predict_layer(outs)
        # logit = logit.masked_fill(mask == 0, -1e9)
        # print(logit)
        # logit = self.predict_layer(outs)

        # list_pos = torch.argmax(logit, dim=2)
        # _, list_pos = torch.topk(logit, k=5)
        _, list_pos = torch.topk(logit, k=20)

        return enc_embs, list_pos



# parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
#                             help='epsilon for label smoothing, 0 means no label smoothing')
# parser.add_argument('--skip', default=1., type=float, metavar='S',
#                     help='pi for skip margin, pi is in the form of probs, must in (0, 1]')
# parser.add_argument('--no-padding', action='store_true',
#                    help='if set, replace pad with unk')
# self.eps = args.label_smoothing
# self.margin = args.skip
# self.pad_idx = 1
# trigger = nn.Embedding(4, 64, max_norm = 1 )
# input = torch.LongTensor([0,1,2,3])
# ifd = trigger(input)
# a = np.random.randint(4,size=4)
# a1 = torch.from_numpy(a).reshape(1,-1).type(torch.int64)
# a1 = torch.from_numpy(np.array([3,2,1,0])).reshape(1,-1).type(torch.int64)
# target = a1
# target = torch.stack((a1,a1,a1,a1)).type(torch.int64)
# predict = torch.rand(4, 10,10)
# b = torch.rand((2, 6, 6))
# c = torch.from_numpy(np.array(list(range(6)))).type(torch.long)
# a = torch.rand((2, 1, 1))
# c.expand_as(b).size()
# predict = torch.rand(1, 4, 4)
# bs, seq_len = target.size()
# target1 = target.repeat(1, seq_len).view(bs, seq_len, seq_len)
#
#
# bipart_no_pad = target1.ne(-1)
# bipart_lprobs = get_normalized_probs(predict)
#
# nll_loss = -bipart_lprobs.gather(dim=-1, index=target1)  # bs seq seq
# nll_loss = nll_loss * bipart_no_pad
#
#
# best_match = np.repeat(np.arange(seq_len).reshape(1, -1, 1), bs, axis=0)  # np.zeros((bs, seq_len, 1))
# nll_loss_numpy = nll_loss.detach().numpy()
# bipart_no_pad[0,0].sum()
# for batch_id in range(bs):
#     no_pad_num = bipart_no_pad[batch_id, 0].sum()
#     raw_index, col_index = lsa(nll_loss_numpy[batch_id, :no_pad_num, :no_pad_num])
#     best_match[batch_id, :no_pad_num] = col_index.reshape(-1, 1)
# best_match = torch.Tensor(best_match).to(target1).long()
#
# nll_loss = nll_loss.gather(dim=-1, index=best_match)
# nll_loss = nll_loss.squeeze(-1)
# nll_loss[nll_loss > 1.0 ] = 0
# epsilon = 0.1
# eps_i = epsilon / bipart_lprobs.size(-1)
# nll_loss = (1 - epsilon) * nll_loss.sum() + eps_i * smooth_loss.sum()
# 带pad命名的变量与文中所提两种策略有关，请详读


# import numpy as np
# a = torch.from_numpy(np.array([0,1,2])).float()
# b = torch.from_numpy(np.array([1,3,6])).float()
# a = torch.randn(1024,20)
# _, indices = torch.topk(a, 4, dim=1)
# indices.detach().numpy().tolist()
# predict = torch.rand(4, 20,1)
# target = torch.rand(4, 20,1)
#
# c = nn.Softmax(dim=1)
# a1 = c(predict).squeeze(2)
# a1 = c(a)
# b1 = c(target).squeeze(2)
# b1 = c(b)
# a1 = a1.reshape(1,-1)
# b1 = b1.reshape(1,-1)
# ptloss2 = torch.nn.NLLLoss(reduce=False)(torch.nn.LogSoftmax(dim=-1)(b1), a1)
# ptloss = F.binary_cross_entropy(b1,a1)
#
# criterion = torch.nn.KLDivLoss()
# klloss = criterion(a1,b1)
# import math
# b1.size()
# len(b1)
# b1[0]


# lo = cross_entropy(b1, a1)
# lo1 = cross_entropy_sep(b1[0], a1[0])
# lo2 = cross_entropy_sep(b1[1], a1[1])
# lo3 = cross_entropy_sep(b1[2], a1[2])
# lo4 = cross_entropy_sep(b1[3], a1[3])
# lo1 + lo2 + lo3 + lo4


# crt = softmax_loss(b1,a1)
# loss = crt(b1,a1)
class GMF(nn.Module):
    def __init__(self, num_user, num_item, mf_dim=10, trainable=True):
        super().__init__()
        self.trainable = trainable
        self.mf_user_emb = nn.Embedding(num_embeddings=num_user, embedding_dim=mf_dim)
        self.mf_item_emb = nn.Embedding(num_embeddings=num_item, embedding_dim=mf_dim)
        if trainable:  # 预训练
            self.linear = nn.Sequential(
                nn.Linear(mf_dim, 1),
                # nn.Sigmoid()
            )
        else:
            trained = torch.load('C:/Users/YWC/Desktop/NeuralCollaborativeFiltering/weights/GMF.pt').state_dict()
            for name, val in self.named_parameters():
                val.data = trained[name]
                val.requires_grad = False

    def forward(self, user_id, item_id):
        mf_vec = self.mf_user_emb(user_id) * self.mf_item_emb(item_id)
        if self.trainable:
            pred = self.linear(mf_vec)
            return pred.squeeze()
        else:
            return mf_vec


class MLP(nn.Module):
    def __init__(self, num_user, num_item, mlp_layers=None, trainable=True):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [10]
        self.trainable = trainable
        self.mlp_user_emb = nn.Embedding(num_embeddings=num_user, embedding_dim=mlp_layers[0] // 2)
        self.mlp_item_emb = nn.Embedding(num_embeddings=num_item, embedding_dim=mlp_layers[0] // 2)
        self.mlp = nn.ModuleList()
        for i in range(1, len(mlp_layers)):
            self.mlp.append(nn.Linear(mlp_layers[i - 1], mlp_layers[i]))
            self.mlp.append(nn.ReLU())
        if trainable:
            self.linear = nn.Sequential(
                nn.Linear(mlp_layers[-1], 1),
                # nn.Sigmoid()
            )
        else:
            trained = torch.load('C:/Users/YWC/Desktop/NeuralCollaborativeFiltering/weights/MLP.pt').state_dict()
            for name, val in self.named_parameters():
                val.data = trained[name]
                val.requires_grad = False

    def forward(self, user_id, item_id):
        mlp_vec = torch.cat([self.mlp_user_emb(user_id), self.mlp_item_emb(item_id)], dim=-1)
        for layer in self.mlp:
            mlp_vec = layer(mlp_vec)
        if self.trainable:
            prediction = self.linear(mlp_vec)
            return prediction.squeeze()
        else:
            return mlp_vec


class NeuMF(nn.Module):

    def __init__(self, num_user, num_item, mf_dim=10,mlp_layers=None, use_pretrain=True):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [10]
        self.gmf = GMF(num_user, num_item, mf_dim, trainable=not use_pretrain)  # 默认直接使用预训练好的权重
        self.mlp = MLP(num_user, num_item, mlp_layers=mlp_layers, trainable=not use_pretrain)
        self.linear = nn.Sequential(
            nn.Linear(mf_dim + mlp_layers[-1], 1),
            # nn.Sigmoid()
        )

    def forward(self, user_id, item_id, uid):
        gmf_vec = self.gmf(user_id, item_id)
        # print(gmf_vec.shape)
        mlp_vec = self.mlp(uid, item_id)
        # NueMF
        cat = torch.cat([gmf_vec, mlp_vec], dim=-1)
        prediction = self.linear(cat)
        return prediction.squeeze()

# input = torch.rand(size=(4,2,4))
# pool=nn.AvgPool2d(kernel_size=(2,1))
# pool=nn.MaxPool2d(kernel_size=(2,1))
# check=pool(input)
# # check