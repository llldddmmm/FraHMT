import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl.nn as dglnn

# class Encoder(nn.Module):
#     def __init__(self, input_size, embed_size,
#                  hidden_size, hidden_layers, latent_size,
#                  dropout, use_gpu):
#         super().__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.hidden_layers = hidden_layers
#         self.latent_size = latent_size
#         self.use_gpu = use_gpu

#         self.rnn = nn.GRU(
#             input_size=self.embed_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.hidden_layers,
#             dropout=dropout,
#             batch_first=True)

#         self.rnn2mean = nn.Linear(
#             in_features=self.hidden_size * self.hidden_layers,
#             out_features=self.latent_size)

#         self.rnn2logv = nn.Linear(
#             in_features=self.hidden_size * self.hidden_layers,
#             out_features=self.latent_size)

#     def forward(self, inputs, embeddings, lengths):
#         batch_size = inputs.size(0)
#         state = self.init_state(dim=batch_size)
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
#         _, state = self.rnn(packed, state)
#         state = state.view(batch_size, self.hidden_size * self.hidden_layers)
#         mean = self.rnn2mean(state)
#         logv = self.rnn2logv(state)
#         std = torch.exp(0.5 * logv)
#         z = self.sample_normal(dim=batch_size)
#         latent_sample = z * std + mean
#         #潜在向量Z!!!!!
#         return latent_sample, mean, std

#     def sample_normal(self, dim):
#         z = torch.randn((self.hidden_layers, dim, self.latent_size))
#         return Variable(z).cuda() if self.use_gpu else Variable(z)

#     def init_state(self, dim):
#         state = torch.zeros((self.hidden_layers, dim, self.hidden_size))
#         return Variable(state).cuda() if self.use_gpu else Variable(state)

class MyGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs['batch_first'] = True  # 设置 batch_first=True
        super(MyGRU, self).__init__(*args, **kwargs)
        # olddim=self.hidden_size
        # middim=int(self.hidden_size*2)
        # newdim=int(self.hidden_size/2)
        # # self.tliner=nn.Linear(olddim,newdim)
        # self.tliner=nn.Sequential(
        #     nn.Linear(olddim,middim),
        #     nn.ReLU(),
        #     nn.Linear(middim,newdim)
        # )

    def forward(self, input, hx=None):
        # 获取隐藏状态的维度

        # 如果没有传入隐藏状态，则初始化为全零
        if hx is None:
            hx = input.new_zeros((self.num_layers, input.size(0), self.hidden_size), requires_grad=False)
        # print(hx.shape)

        # origin_state = self.tliner(hx)
        origin_state= hx*0.2
        

        # print(origin_state.shape)
        # 按时间步迭代计算
        outputs = []

        # print(input.data.shape)

        #时间步 
        for t in range(input.size(1)):
            # 在每个时间步之前修改隐藏状态
            if t == 0:
                modified_state = hx
            else:
                            #    state=self.tliner(hx)
                modified_state=torch.add(origin_state, hx*0.8)
            #    print(state.shape)
            #    modified_state = torch.cat((origin_state,state),dim=2)  # 示例：隐藏状态向量乘以2

            # 调用父类 GRU 的 forward 方法，传入修改后的隐藏状态向量
            # print(modified_state.shape)
            output, hx = super(MyGRU, self).forward(input[:, t, :].unsqueeze(1), modified_state)
            # print(output.shape)
            # print(hx.shape)

            # 保存当前时间步的输出
            outputs.append(output.squeeze(1))

        # 将输出堆叠成一个张量
        output = torch.stack(outputs, dim=1)

        return output, hx

class Decoder(nn.Module):
    def __init__(self, embed_size, latent_size, hidden_size,
                 hidden_layers, dropout, output_size):
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout

        self.rnn = MyGRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True)

        self.rnn2out = nn.Linear(
            in_features=hidden_size,
            out_features=output_size)

    # def forward(self, embeddings, state, lengths):
    #     batch_size = embeddings.size(0)
    #     packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    #     hidden, state = self.rnn(packed, state)
    #     state = state.view(self.hidden_layers, batch_size, self.hidden_size)
    #     hidden, _ = pad_packed_sequence(hidden, batch_first=True)
    #     output = self.rnn2out(hidden)
    #     return output, state
    def forward(self, embeddings, state, lengths):
        batch_size = embeddings.size(0)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # print(packed.data.shape)
        # print(packed.batch_sizes)
        # print(embeddings.shape)
        hidden, state = self.rnn(embeddings, state)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        # hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        output = self.rnn2out(hidden)
        return output, state

class Frag2Mol(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)

        self.latent2rnn = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size)
        
        encoder_model_args={
		"atom_dim": 42,
		"bond_dim": 14,
		"pharm_dim": 194,
		"reac_dim": 34,
		"hid_dim": 300,
		"depth": 3,
		"act": "ReLU",
		"num_task": 1
        }
        sssmodel = PharmHGT(encoder_model_args)
        self.encoder=sssmodel
        
        self.decoder = Decoder(
            embed_size=self.embed_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            output_size=self.input_size)

    def forward(self, inputs, lengths, sss):
        batch_size = inputs.size(0)
        embeddings = self.embedder(inputs)
        # embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training)
        z, mu, sigma = self.encoder(sss)
        state = self.latent2rnn(z)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        return output, mu, sigma

    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()


class Loss(nn.Module):
    def __init__(self, config, pad):
        super().__init__()
        self.config = config
        self.pad = pad

    def forward(self, output, target, mu, sigma, epoch):
        output = F.log_softmax(output, dim=1)

        # flatten all predictions and targets
        target = target.view(-1)
        output = output.view(-1, output.size(2))

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        output = output[range(output.size(0)), target] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        CE_loss = -torch.sum(output) / nb_tokens

        # compute KL Divergence
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss
        return CE_loss + KL_loss




import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from functools import partial
import copy

import math




from torch import nn
import numpy as np
import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss

def remove_nan_label(pred,truth):
    #去除预测值和真实值中的NAN（异常值）
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth

def roc_auc(pred,truth):
    #计算roc曲线
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    #计算均方根误差
    return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    #计算平均绝对误差
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           }

def get_func(fn_name):
    fn_name = fn_name.lower()
    #将字符转化为小写
    return func_dict[fn_name]
#返回所需要的函数



# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]
#将tensor数据每两个交换位置

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}
#返回边的源节点的field特征 减去 边特征

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}
#返回原子特征,一行一行的


# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#返回N个相同的module层

def attention(query, key, value, mask=None, dropout=None):
    #维度为 N H L_q/k/v D
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #表示每个q对每个k的相似度分数
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    #表示每个q对每个k的注意力权重
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)  # 不影响
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
#返回一个输出张量（注意力权重*value）和一个注意力权重    维度均为N H L_q L_k

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        #用zip进行打包，l是线性层，x是tensor，使用l（x）得到 形状 N H L D的数据给q k v-----通过线性变换将数据映射到不同的子空间，用于多头注意力机制

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #将多个头的注意力输出拼接起来？
        return self.linears[-1](x)

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6,hid_dim)
        #六头的注意力机制
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
        #双向gru
    
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]  #获取bg（一个批次的异构图数据）中指定的节点类型和字段的数据
        node_size = bg.batch_num_nodes(ntype)  #获取指定节点类型的数量
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        #计算每个图中指定节点类型的起始索引？？

        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            #截取第i个异构图的数据
            # print(cur_hidden.shape)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            #在四个维度上进行补零操作？
            # print(cur_hidden.shape)
            # print(cur_hidden.unsqueeze(0).shape)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        # print(hidden_lst.shape)
        #将所有张量沿着第0维度拼接起来，得到一个四维张量（批次大小、最大节点数、隐藏层维度、方向数）

        return hidden_lst
        
    def forward(self,bg,suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        # print(bg.nodes['a'].data)
        # print(bg.nodes['a'].data['f'].shape)
        # print(bg.nodes['p'].data['f'].shape)
        # print(bg.nodes['p'].data['f_h'].shape)
        # print(bg.nodes['p'].data['f_aug'].shape)
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        #片段数据
        # print(p_pharmj.shape)

        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)
        #原子数据
        # print(a_pharmj.shape)
        # print(bg.nodes['a'].data['f'].shape)
        # print(bg.nodes['p'].data['f'].shape)

        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        #使用matmul方法进行矩阵乘法，得到一个二维张量，表示每个’a’节点和每个’p’节点之间是否有连接。得到一个掩码张量，赋值给mask

        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj,mask) + a_pharmj
        #？对’a’节点和’p’节点进行多头注意力操作，并将结果与’a’节点的原始特征相加，得到一个三维张量，每一行对应一个子图中’a’节点的注意力加权后的特征向量，并赋值给一个变量h

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        #？对’h’张量进行最大池化操作，并增加一个维度，并重复两次（因为是双向GRU），得到一个三维张量，每一行对应一个子图中’a’节点的最大特征向量，并赋值给一个变量hidden
        h, hidden = self.gru(h, hidden)
        #对’h’张量和’hidden’张量进行双向GRU操作，得到两个三维张量，分别表示每个子图中’a’节点的输出特征向量和最终状态向量
        
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)
        #从h里面获取  g的图级表示

        return graph_embed
       
# class MVMP(nn.Module):
#     #直接在输入的异构图bg上进行操作，三轮消息传递及更新
#     def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU()):
#         """
#         MultiViewMassagePassing
#         view: a, ap, apj
#         suffix: filed to save the nodes' hidden state in dgl.graph. 
#                 e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
#         """
#         super(MVMP,self).__init__()
#         self.view = view
#         self.depth = depth
#         self.suffix = suffix
#         self.msg_func = msg_func
#         self.act = act
#         self.homo_etypes = [('a','b','a')]
#         self.hetero_etypes = []
#         self.node_types = ['a','p']
#         if 'p' in view:
#             self.homo_etypes.append(('p','r','p'))
#         if 'j' in view:
#             self.node_types.append('junc')
#             self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

#         self.attn = nn.ModuleDict()
#         for etype in self.homo_etypes + self.hetero_etypes:
#             self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

#         self.mp_list = nn.ModuleDict()
#         for edge_type in self.homo_etypes:
#             self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

#         self.node_last_layer = nn.ModuleDict()
#         for ntype in self.node_types:
#             self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

#     def update_edge(self,edge,layer):
#         return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
#     def update_node(self,node,field,layer):
#         return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
#                                        node.data[field],
#                                        node.data['f']],1))}
#     def init_node(self,node):
#         return {f'f_{self.suffix}':node.data['f'].clone()}

#     def init_edge(self,edge):
#         return {'h':edge.data['x'].clone()}


#     def forward(self,bg):
#         suffix = self.suffix
#         for ntype in self.node_types:
#             if ntype != 'junc':
#                 bg.apply_nodes(self.init_node,ntype=ntype)
#         for etype in self.homo_etypes:
#             bg.apply_edges(self.init_edge,etype=etype)

#         if 'j' in self.view:
#             bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
#             bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

#         update_funcs = {e:(fn.copy_e('h','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
#         update_funcs.update({e:(fn.copy_src(f'f_junc_{suffix}','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
#         # message passing
#         for i in range(self.depth-1):
#             bg.multi_update_all(update_funcs,cross_reducer='sum')
#             for edge_type in self.homo_etypes:
#                 bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
#                 bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
#                 bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

#         # last update of node feature
#         update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
#         bg.multi_update_all(update_funcs,cross_reducer='sum')

#         # last update of junc feature
#         bg.multi_update_all({e:(fn.copy_src(f'f_junc_{suffix}','mail'),
#                                  partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
#                                  cross_reducer='sum')



class HeteroGraphConvNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HeteroGraphConvNet, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'b': dglnn.GraphConv(in_feats, hidden_feats),
            'r': dglnn.GraphConv(in_feats, hidden_feats),
            'j': dglnn.GraphConv(in_feats, hidden_feats),
            'j': dglnn.GraphConv(in_feats, hidden_feats)
        })
        self.conv2 = dglnn.HeteroGraphConv({
            'b': dglnn.GraphConv(hidden_feats, out_feats),
            'r': dglnn.GraphConv(hidden_feats, out_feats),
            'j': dglnn.GraphConv(hidden_feats, out_feats),
            'j': dglnn.GraphConv(hidden_feats, out_feats)
        })

    def forward(self, g, node_feats):
        node_feats = self.conv1(g, node_feats)
        node_feats = {k: torch.relu(v) for k, v in node_feats.items()}
        node_feats = self.conv2(g, node_feats)
        return node_feats

class PharmHGT(nn.Module):
    def __init__(self,args):
        super(PharmHGT,self).__init__()

        self.use_gpu=False
        hid_dim = args['hid_dim']
        self.act = get_func(args['act'])
        self.depth = args['depth']

        self.w_atom = nn.Linear(args['atom_dim'],hid_dim)
        self.w_bond = nn.Linear(args['bond_dim'],hid_dim)
        self.w_pharm = nn.Linear(args['pharm_dim'],hid_dim)
        self.w_reac = nn.Linear(args['reac_dim'],hid_dim)
        self.w_junc = nn.Linear(args['atom_dim'] + args['pharm_dim'],hid_dim)


        self.readout = Node_GRU(hid_dim)
        self.readout_attn = Node_GRU(hid_dim)
        self.initialize_weights()
        self.qliner=nn.Linear(600, 64)
        self.gcn2=nn.Linear(600,128)

        self.rnn2mean = nn.Linear(
            in_features=64 * 2,
            out_features=100)

        self.rnn2logv = nn.Linear(
            in_features=64 * 2,
            out_features=100)
        
        in_feats = 300
        hidden_feats = 100
        out_feats = 20
        self.GCNmodel = HeteroGraphConvNet(in_feats, hidden_feats, out_feats)

        self.pliner=nn.Linear(240,128)
        self.aliner=nn.Linear(760,128)

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self,bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))

    def sample_normal(self, dim):
        z = torch.randn((2, dim, 100))
        return Variable(z).cuda() if self.use_gpu else Variable(z)        

    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]  
        node_size = bg.batch_num_nodes(ntype)  
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        if ntype == 'a':
            max_num_node = 38
        else:
            max_num_node = 12
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst

    def forward(self,bg):
        """
        Args:
            bg: a batch of graphs
        """
        self.init_feature(bg)

        output = self.GCNmodel(bg, {'a':bg.nodes['a'].data['f'] , 'p':bg.nodes['p'].data['f']})
        bg.nodes['a'].data['f_h']=output['a']
        bg.nodes['p'].data['f_h']=output['p']
        device = bg.device
        p_pharmj = self.split_batch(bg,'p',f'f_h',device)
        p_pharmj=p_pharmj.reshape(32, -1)
        a_pharmj = self.split_batch(bg,'a',f'f_h',device)
        a_pharmj=a_pharmj.reshape(32, -1)
        p_emb=self.pliner(p_pharmj)
        a_emb=self.aliner(a_pharmj)
        embed=torch.add(p_emb,a_emb)

        state=embed
        mean = self.rnn2mean(state)
        logv = self.rnn2logv(state)
        std = torch.exp(0.5 * logv)
        z = self.sample_normal(dim=32)
        z=z.cuda()
        latent_sample = z * std + mean
        return latent_sample, mean, std
        
