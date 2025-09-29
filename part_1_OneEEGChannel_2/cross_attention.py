#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cross_attention.py

Cross Attention module for audio–EEG fusion.
基于 DepthConv1d + ScaledDotProductAttention + 多层交替 cross attention。
主要功能：融合 audio 和 EEG 的特征表示。
"""

import torch
import torch.nn as nn

IDX2LABEL = ["Gt", "Vx", "Dr", "Bs"]

def auto_transpose(x):
    """
    自动检测输入格式:
    - 如果输入是 [B, C, T]，转成 [B, T, C]
    - 如果输入是 [B, T, C]，保持不变
    """
    B, dim2, dim3 = x.shape
    if dim2 < dim3:   # 通道数通常小于时间长度 → [B, C, T]
        return x.transpose(1, 2)   # 转换为 [B, T, C]
    else:             # 已经是 [B, T, C]
        return x


# ============ 基础模块 ============
class ScaledDotProductAttention(nn.Module):
    """
    标准缩放点积注意力 (Scaled Dot-Product Attention)
    q: [B, n_head, Lq, d_k]
    k: [B, n_head, Lk, d_k]
    v: [B, n_head, Lv, d_v]
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 缩放因子，通常 = sqrt(d_k)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # 计算注意力分数 [B, n_head, Lq, Lk]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # softmax 正则化得到注意力权重
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和，得到输出 [B, n_head, Lq, d_v]
        output = torch.matmul(attn, v)

        return output, attn


class DepthConv1d(nn.Module):
    """
    Depthwise + Pointwise 1D 卷积
    用于 Q/K/V 的投影。
    """
    def __init__(self, kernel, input_channel, hidden_channel, dilation=1, padding='same'):
        super().__init__()
        # 深度卷积：每个通道单独卷积
        self.depthwise = nn.Conv1d(input_channel, input_channel, kernel_size=kernel,
                                   groups=input_channel, dilation=dilation, padding=padding)
        # 逐点卷积：整合通道
        self.pointwise = nn.Conv1d(input_channel, hidden_channel, kernel_size=1)

    def forward(self, x):
        """
        输入: [B, T, C]  → 转换成 [B, C, T] 方便 Conv1d
        输出: [B, T, hidden_channel]
        """
        x = x.transpose(1, 2)              # [B, C, T]
        out = self.depthwise(x)            # [B, C, T]
        out = self.pointwise(out)          # [B, hidden, T]
        out = out.transpose(1, 2)          # [B, T, hidden]
        return out, None


# ============ Cross Attention 层 ============
class ConvCrossAttention(nn.Module):
    """
    基于卷积的 Cross Attention
    - 输入: [B, T, C]
    - 输出: [B, T, C]
    """
    def __init__(self, n_head, d_model, d_k, d_v, in_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 使用卷积生成 Q/K/V 表示
        self.w_qs = DepthConv1d(kernel=kernel_size, input_channel=in_ch,
                                hidden_channel=n_head * d_k,
                                dilation=dilation, padding='same')
        self.w_ks = DepthConv1d(kernel=kernel_size, input_channel=in_ch,
                                hidden_channel=n_head * d_k,
                                dilation=dilation, padding='same')
        self.w_vs = DepthConv1d(kernel=kernel_size, input_channel=in_ch,
                                hidden_channel=n_head * d_v,
                                dilation=dilation, padding='same')

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.GroupNorm(1, in_ch, eps=1e-08)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [B, T, C]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v  # 残差连接

        # Q, K, V 卷积投影
        q, _ = self.w_qs(q)  # [B, T, n_head*d_k]
        k, _ = self.w_ks(k)
        v, _ = self.w_vs(v)

        # reshape 为多头形式
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # 转置以适配注意力计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展到多头维度

        # 注意力计算
        out, attn = self.attention(q, k, v, mask=mask)

        # 合并多头结果
        out = out.transpose(1, 2).contiguous().view(sz_b, len_v, -1)  # [B, L, C]
        out = self.dropout(out)

        # 残差 + 归一化
        out += residual
        out = out.transpose(1, 2)              # [B, C, L]
        out = self.layer_norm(out)
        out = out.transpose(1, 2)              # [B, L, C]

        return out


# ============ 多层交替 Cross Attention ============
class MultiLayerCrossAttention(nn.Module):
    def __init__(self, audio_in_ch, eeg_in_ch, hidden_ch, layer, kernel_size, dilation):
        super().__init__()
        self.layer = layer

        # 用 Linear 把 audio / EEG 投影到相同 hidden_ch
        self.audio_proj = nn.Linear(audio_in_ch, hidden_ch)
        self.eeg_proj   = nn.Linear(eeg_in_ch, hidden_ch)

        self.audio_encoder = nn.ModuleList()
        self.eeg_encoder = nn.ModuleList()
        self.LayernormList_audio = nn.ModuleList()
        self.LayernormList_eeg = nn.ModuleList()

        self.projection = nn.Conv1d(hidden_ch * 4, hidden_ch, kernel_size, padding='same')
        self.layernorm_out = nn.GroupNorm(1, hidden_ch, eps=1e-08)

        for _ in range(layer):
            self.audio_encoder.append(
                ConvCrossAttention(n_head=1, d_model=hidden_ch, d_k=hidden_ch, d_v=hidden_ch,
                                   in_ch=hidden_ch, kernel_size=kernel_size, dilation=dilation)
            )
            self.eeg_encoder.append(
                ConvCrossAttention(n_head=1, d_model=hidden_ch, d_k=hidden_ch, d_v=hidden_ch,
                                   in_ch=hidden_ch, kernel_size=kernel_size, dilation=dilation)
            )
            self.LayernormList_audio.append(nn.GroupNorm(1, hidden_ch, eps=1e-08))
            self.LayernormList_eeg.append(nn.GroupNorm(1, hidden_ch, eps=1e-08))

    def forward(self, audio, eeg):
        # 保证是 [B, T, C]
        audio = auto_transpose(audio)
        eeg = auto_transpose(eeg)

        # 投影到相同 hidden_ch
        audio = self.audio_proj(audio)
        eeg = self.eeg_proj(eeg)

        out_audio, out_eeg = audio, eeg
        skip_audio, skip_eeg = 0., 0.
        residual_audio, residual_eeg = audio, eeg

        for i in range(self.layer):
            out_audio = self.audio_encoder[i](out_eeg, out_audio, out_audio)
            out_eeg = self.eeg_encoder[i](out_audio, out_eeg, out_eeg)

            # audio norm
            out_audio = (out_audio + residual_audio).transpose(1, 2)
            out_audio = self.LayernormList_audio[i](out_audio)
            out_audio = out_audio.transpose(1, 2)

            # eeg norm
            out_eeg = (out_eeg + residual_eeg).transpose(1, 2)
            out_eeg = self.LayernormList_eeg[i](out_eeg)
            out_eeg = out_eeg.transpose(1, 2)

            residual_audio, residual_eeg = out_audio, out_eeg
            skip_audio += out_audio
            skip_eeg += out_eeg

        out = torch.cat((skip_audio, audio, skip_eeg, eeg), dim=2)  # [B, T, 4*hidden_ch]
        out = out.transpose(1, 2)  # [B, 4*hidden_ch, T]
        out = self.projection(out) # [B, hidden_ch, T]
        out = self.layernorm_out(out)
        return out

class CrossAttentionClassifier(nn.Module):
    def __init__(self, audio_in_ch, eeg_in_ch, hidden_ch, num_classes, layer=2, kernel_size=3, dilation=1):
        super().__init__()
        # 复用 MultiLayerCrossAttention
        self.cross_att = MultiLayerCrossAttention(
            audio_in_ch=audio_in_ch,
            eeg_in_ch=eeg_in_ch,
            hidden_ch=hidden_ch,
            layer=layer,
            kernel_size=kernel_size,
            dilation=dilation
        )
        # 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)   # [B, C, T] → [B, C, 1]
        self.fc = nn.Linear(hidden_ch, num_classes)

    def forward(self, audio, eeg):
        feats = self.cross_att(audio, eeg)     # [B, C, T]
        pooled = self.pool(feats).squeeze(-1)  # [B, C]
        logits = self.fc(pooled)               # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)  # 概率分布
        return probs

def get_predicted_labels(probs: torch.Tensor):
    """
    根据 cross attention 输出的概率分布，返回预测的类别标签

    参数:
        probs: [B, 4] 张量，已经是 softmax 概率

    返回:
        pred_idx: [B] 张量，类别索引
        pred_labels: [B] 列表，类别字符串
    """
    pred_idx = torch.argmax(probs, dim=1)  # [B]
    pred_labels = [IDX2LABEL[i] for i in pred_idx.tolist()]
    return pred_idx, pred_labels


# ============ 测试 ============
if __name__ == "__main__":
    B, T_a, C_a = 4, 5000, 2   # audio
    B, T_e, C_e = 4, 5000, 20  # EEG
    audio_feats = torch.randn(B, T_a, C_a)
    eeg_feats   = torch.randn(B, T_e, C_e)

    model = CrossAttentionClassifier(audio_in_ch=2, eeg_in_ch=20, hidden_ch=64, num_classes=4)
    out = model(audio_feats, eeg_feats)

    pred_idx, pred_labels = get_predicted_labels(out)
    print("预测类别索引:", pred_idx.tolist())
    print("预测类别标签:", pred_labels)

    print(out.shape)  # [4, 4] -> 每个样本对 4 个乐器的概率
    print(out)        # 比如 [0.1, 0.7, 0.1, 0.1] 表示预测是 "Gt"
