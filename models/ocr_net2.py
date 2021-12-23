from torch import nn
from torchvision.models import resnet18
import torch
from einops import rearrange


class SelfAttention(nn.Module):
    r"""多头自注意力

    Args:
        embed_dim: 词向量的特征数。
        num_head: 多头注意力的头数。
        is_mask: 是否添加掩码。是，则网络只能看到每个词前的内容，而无法看到后面的内容。

    Shape:
        - Input: N,S,V (批次，序列数，词向量特征数)
        - Output:same shape as the input

    Examples::
        # >>> m = SelfAttention(720, 12)
        # >>> x = torch.randn(4, 13, 720)
        # >>> output = m(x)
        # >>> print(output.shape)
        # torch.Size([4, 13, 720])
    """

    def __init__(self, embed_dim, num_head, is_mask=True):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.is_mask = is_mask
        self.linear1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        '''x 形状 N,S,V'''
        x = self.linear1(x)  # 形状变换为N,S,3V
        n, s, v = x.shape
        """分出头来,形状变换为 N,S,H,V"""
        x = x.reshape(n, s, self.num_head, -1)
        """换轴，形状变换至 N,H,S,V"""
        x = torch.transpose(x, 1, 2)
        '''分出Q,K,V'''
        query, key, value = torch.chunk(x, 3, -1)
        dk = value.shape[-1] ** 0.5
        '''计算自注意力'''
        w = torch.matmul(query, key.transpose(-1, -2)) / dk  # w 形状 N,H,S,S
        if self.is_mask:
            """生成掩码"""
            mask = torch.tril(torch.ones(w.shape[-1], w.shape[-1])).to(w.device)
            w = w * mask - 1e10 * (1 - mask)
        w = torch.softmax(w, dim=-1)  # softmax归一化
        attention = torch.matmul(w, value)  # 各个向量根据得分合并合并, 形状 N,H,S,V
        '''换轴至 N,S,H,V'''
        attention = attention.permute(0, 2, 1, 3)
        n, s, h, v = attention.shape
        '''合并H，V，相当于吧每个头的结果cat在一起。形状至N,S,V'''
        attention = attention.reshape(n, s, h * v)
        return self.linear2(attention)  # 经过线性层后输出


class Block(nn.Module):
    r"""

    Args:
        embed_dim: 词向量的特征数。
        num_head: 多头注意力的头数。
        is_mask: 是否添加掩码。是，则网络只能看到每个词前的内容，而无法看到后面的内容。

    Shape:
        - Input: N,S,V (批次，序列数，词向量特征数)
        - Output:same shape as the input

    Examples::
        # >>> m = Block(720, 12)
        # >>> x = torch.randn(4, 13, 720)
        # >>> output = m(x)
        # >>> print(output.shape)
        # torch.Size([4, 13, 720])
    """

    def __init__(self, embed_dim, num_head, is_mask):#512，8，false
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)# channel 方向做归一化，算chw均值，rnn中作用明显
        self.attention = SelfAttention(embed_dim, num_head, is_mask)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 6),
            nn.ReLU(),
            nn.Linear(embed_dim * 6, embed_dim)
        )

    def forward(self, x):
        '''计算多头自注意力'''
        attention = self.attention(self.ln_1(x))
        '''残差'''
        x = attention + x
        x = self.ln_2(x)
        '''计算feed forward部分'''
        h = self.feed_forward(x)
        x = h + x  # 增加残差
        return x


class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,#(3,9)
        dim_head#512
    ):
        super().__init__()
        height, width = fmap_size # 3,9
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (w h) d')
        # logits = torch.einsum('b i d, j d -> b i j', q, emb)
        return emb


class OcrNet(nn.Module):
    """input--image:[3,48,144],target:[15]"""
    """output--[27, b, num_class]"""
    def __init__(self,num_class):
        super(OcrNet, self).__init__()
        resnet = resnet18(True)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            *backbone[:3],
            *backbone[4:8],
        )
        self.decoder = nn.Sequential(
            Block(512, 8, False),
            Block(512, 8, False),
            Block(512, 8, False),
        )
        self.out_layer = nn.Linear(512, num_class)
        self.abs_pos_emb = AbsPosEmb((3, 9), 512)
        # print(self.backbone)

    def forward(self,x):
        x = self.backbone(x)
        x = rearrange(x,'n c h w -> n (w h) c')
        x = x + self.abs_pos_emb()
        x = self.decoder(x)
        x = rearrange(x, 'n s v -> s n v')
        return self.out_layer(x)

if __name__ == '__main__':
    m = OcrNet(70)
    print(m)
    x = torch.randn(32,3,48,144)
    print(m(x).shape)

