import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp

#
# 步骤 1: 构建标准模块 (MLP, PatchEmbed, Downsample)
#

class PatchEmbed(nn.Module):
    """ 图像到Patch的嵌入 """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用单个Conv2d实现 s=4 的 Patch 嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) # (B, C_embed, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (B, N, C_embed)
        x = self.norm(x)
        H_out, W_out = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H_out, W_out)

class Downsample(nn.Module):
    """ 阶段之间的下采样模块 """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_embed_dim)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x) # (B, C_out, H/2, W/2)
        x = x.flatten(2).transpose(1, 2) # (B, N/4, C_out)
        x = self.norm(x)
        H_out, W_out = H // 2, W // 2
        return x, (H_out, W_out)

#
# 步骤 2: 实现核心的 Deformable Attention 模块
# (基于 DAT++ 论文, Fig 2 )
#

class DeformableAttention(nn.Module):
    """
    Deformable Multi-Head Attention (DMHA)
    参数基于 DATA Table I  和 DAT++ Table 1 [cite: 1148] 推断.
    """
    def __init__(self, dim, num_heads, grid_size=7, offset_groups=1, use_rpb=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = to_2tuple(grid_size) # H_G, W_G (e.g., 7x7)
        self.offset_groups = offset_groups
        self.num_sampling_points = self.grid_size[0] * self.grid_size[1]
        self.scale = self.head_dim ** -0.5
        self.use_rpb = use_rpb

        # 1. Q, K, V 线性投射
        self.q_proj = nn.Linear(dim, dim)
        # K, V 的投影：每个offset group对应部分heads
        # 但为了实现简单，我们仍然投影到完整的dim，后续通过reshape分配
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # 2. Offset 生成网络 (如 DAT++ Fig 2b) [cite: 961]
        # 我们使用一个DWC捕获局部信息 [cite: 969]，
        # 然后用自适应池化来处理动态的stride 'r'
        self.offset_local_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.offset_pool = nn.AdaptiveAvgPool2d(self.grid_size)
        self.offset_norm = nn.LayerNorm(dim)
        self.offset_gelu = nn.GELU()
        # 输出 G * 2 个偏移量 (x, y) [cite: 935]
        self.offset_conv = nn.Conv2d(dim, offset_groups * 2, kernel_size=1, bias=False)

        # 3. 参考点 (Reference Points) [cite: 991]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, self.grid_size[0] - 0.5, self.grid_size[0]),
            torch.linspace(0.5, self.grid_size[1] - 0.5, self.grid_size[1])
        )
        # 归一化到 [0, 1]
        ref_y = ref_y / self.grid_size[0]
        ref_x = ref_x / self.grid_size[1]
        # 转换到 [-1, 1] 以便 F.grid_sample
        ref_grid = torch.stack((ref_x, ref_y), -1).view(self.num_sampling_points, 2)
        ref_grid = (ref_grid * 2.0 - 1.0).unsqueeze(0).unsqueeze(2) # (1, Ns, 1, 2)
        self.register_buffer("reference_points", ref_grid)

        # 4. 相对位置偏置 (RPB)
        # 我们使用 DATA 论文 [cite: 156] 图中所示的简化版本
        # 偏置仅取决于采样点的位置
        if self.use_rpb:
            self.rpb_table = nn.Parameter(
                torch.zeros(self.num_heads, self.grid_size[0] * 2 - 1, self.grid_size[1] * 2 - 1)
            )
            trunc_normal_(self.rpb_table, std=.02)

            # 用于RPRB采样
            coords_y, coords_x = torch.meshgrid(
                torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1])
            )
            coords = torch.stack([coords_y, coords_x], dim=-1).float()
            coords = coords.view(self.num_sampling_points, 2)
            self.register_buffer("rpb_coords", coords)


    def _sample_rpb(self, deformed_points):
        """ 
        根据 DATA Fig 2 [cite: 156] (Attn bias sample) 采样RPE Bias
        
        参数:
            deformed_points: (B, Ns, G, 2) 变形后的采样点坐标，范围 [-1, 1]
        
        返回:
            rpb: (B, M, 1, Ns) 相对位置偏置，用于添加到attention logits
        
        维度说明:
            B: batch size
            Ns: 采样点数量 (grid_size[0] * grid_size[1])
            G: offset groups
            M: num_heads
            rpb_table: (M, 2*H_G-1, 2*W_G-1) 可学习的偏置表
        """
        B = deformed_points.shape[0]
        
        # 步骤1: 重塑采样坐标以便批量处理
        # (B, Ns, G, 2) -> (B, G, Ns, 2) -> (B*G, Ns, 2)
        sample_coords = deformed_points.transpose(1, 2).reshape(
            B * self.offset_groups, self.num_sampling_points, 2
        )
        
        # 步骤2: 对每个head组进行grid_sample
        # rpb_table: (M, 2H-1, 2W-1)
        # 需要扩展为 (B*G, M//G, 2H-1, 2W-1) 以便批量采样
        heads_per_group = self.num_heads // self.offset_groups
        
        # 为每个offset group重复对应的heads
        # (M, 2H-1, 2W-1) -> (G, M//G, 2H-1, 2W-1)
        rpb_table_grouped = self.rpb_table.view(
            self.offset_groups, heads_per_group, 
            self.rpb_table.shape[1], self.rpb_table.shape[2]
        )
        # (G, M//G, 2H-1, 2W-1) -> (1, G, M//G, 2H-1, 2W-1) -> (B, G, M//G, 2H-1, 2W-1)
        rpb_table_expanded = rpb_table_grouped.unsqueeze(0).expand(
            B, -1, -1, -1, -1
        )
        # (B, G, M//G, 2H-1, 2W-1) -> (B*G, M//G, 2H-1, 2W-1)
        rpb_table_batched = rpb_table_expanded.reshape(
            B * self.offset_groups, heads_per_group,
            self.rpb_table.shape[1], self.rpb_table.shape[2]
        )
        
        # 步骤3: 使用grid_sample采样RPB
        # sample_coords: (B*G, Ns, 2) -> (B*G, 1, Ns, 2) for grid_sample
        # rpb_table_batched: (B*G, M//G, 2H-1, 2W-1)
        rpb = F.grid_sample(
            rpb_table_batched,  # (B*G, M//G, 2H-1, 2W-1)
            sample_coords.unsqueeze(1),  # (B*G, 1, Ns, 2)
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # 输出: (B*G, M//G, 1, Ns)
        
        # 步骤4: 重塑为最终形状
        # (B*G, M//G, 1, Ns) -> (B, G, M//G, 1, Ns) -> (B, M, 1, Ns)
        rpb = rpb.view(B, self.offset_groups, heads_per_group, 1, self.num_sampling_points)
        rpb = rpb.transpose(1, 2).reshape(B, self.num_heads, 1, self.num_sampling_points)
        
        return rpb


    def forward(self, x, hw_shape):
        """
        前向传播
        
        参数:
            x: (B, N, C) 输入token序列
            hw_shape: (H, W) 空间维度
        
        返回:
            out: (B, N, C) 输出token序列
        
        详细维度流程:
            B: batch size
            N: token数量 (H * W)
            C: 通道数/嵌入维度
            M: num_heads (注意力头数)
            D_h: head_dim (每个头的维度, C // M)
            G: offset_groups (偏移量组数)
            Ns: num_sampling_points (采样点数, grid_size[0] * grid_size[1])
        """
        B, N, C = x.shape
        H, W = hw_shape
        M, D_h = self.num_heads, self.head_dim
        G, Ns = self.offset_groups, self.num_sampling_points

        # 1. 计算 Q: (B, N, C) -> (B, M, N, D_h)
        q = self.q_proj(x)  # (B, N, C)
        q = q.reshape(B, N, M, D_h).permute(0, 2, 1, 3)  # (B, M, N, D_h)

        # 2. 计算 Offsets: (B, N, C) -> (B, C, H, W)
        x_q_for_offset = x.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        
        # 步骤2a: 局部深度卷积提取特征
        # (B, C, H, W) -> (B, C, H, W)
        offset_feat = self.offset_local_conv(x_q_for_offset)
        
        # 步骤2b: 自适应池化到grid_size
        # (B, C, H, W) -> (B, C, H_G, W_G) where H_G=W_G=grid_size
        offset_feat = self.offset_pool(offset_feat)  # (B, C, H_G, W_G)
        
        # 步骤2c: 归一化和激活
        # (B, C, H_G, W_G) -> (B, Ns, C) where Ns = H_G * W_G
        offset_feat = offset_feat.flatten(2).transpose(1, 2)  # (B, Ns, C)
        offset_feat = self.offset_norm(offset_feat)  # (B, Ns, C)
        offset_feat = self.offset_gelu(offset_feat)  # (B, Ns, C)
        
        # 步骤2d: 生成偏移量
        # (B, Ns, C) -> (B, C, H_G, W_G)
        offset_feat = offset_feat.transpose(1, 2).view(B, C, self.grid_size[0], self.grid_size[1])
        # (B, C, H_G, W_G) -> (B, G*2, H_G, W_G)
        offsets = self.offset_conv(offset_feat)  # (B, G*2, H_G, W_G)
        # (B, G*2, H_G, W_G) -> (B, Ns, G, 2)
        offsets = offsets.permute(0, 2, 3, 1).view(B, Ns, G, 2)  # (B, Ns, G, 2)

        # 3. 计算 Deformed Points (变形采样点)
        # reference_points: (1, Ns, 1, 2) 预定义的规则网格 [-1, 1]
        # offsets: (B, Ns, G, 2) 学习到的偏移量
        # (B, Ns, G, 2) + (1, Ns, 1, 2) -> (B, Ns, G, 2)
        deformed_points = (offsets + self.reference_points).tanh()  # 限制在 [-1, 1]
        
        # 4. 双线性插值采样特征
        # 步骤4a: 准备采样坐标
        # (B, Ns, G, 2) -> (B, Ns*G, 2)
        sample_coords = deformed_points.reshape(B, Ns * G, 2)  # (B, Ns*G, 2)
        
        # 步骤4b: 从输入特征图采样
        # x_q_for_offset: (B, C, H, W)
        # sample_coords: (B, 1, Ns*G, 2) grid_sample需要4D坐标
        sampled_features = F.grid_sample(
            x_q_for_offset,  # (B, C, H, W)
            sample_coords.unsqueeze(1),  # (B, 1, Ns*G, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # 输出: (B, C, 1, Ns*G)
        
        # 步骤4c: 重塑采样特征
        # (B, C, 1, Ns*G) -> (B, C, Ns, G)
        sampled_features = sampled_features.view(B, C, Ns, G)  # (B, C, Ns, G)
        # (B, C, Ns, G) -> (B, G, Ns, C)
        sampled_features = sampled_features.permute(0, 3, 2, 1)  # (B, G, Ns, C)
        # (B, G, Ns, C) -> (B*G, Ns, C)
        sampled_features = sampled_features.reshape(B * G, Ns, C)  # (B*G, Ns, C)

        # 5. 计算 K 和 V
        # sampled_features: (B*G, Ns, C)
        # 
        # 理解 offset groups:
        # - 每个 group 独立采样 Ns 个点
        # - 共有 G 个组，每组负责 M//G 个attention头
        # - 因此实际上每组应该有 Ns 个采样点，总共 G*Ns 个不同的采样点
        #
        # 但在当前实现中，G个组共享相同的Ns个采样位置（只是offset不同）
        # 所以我们需要为每个组生成对应的K、V
        
        # 方案: 每个组单独处理，输出对应头的维度
        # (B*G, Ns, C) -> (B*G, Ns, (M//G) * D_h)
        # 
        # 但k_proj是 Linear(C, C)，输出 C = M * D_h
        # 我们需要将输出按组分配
        
        k = self.k_proj(sampled_features)  # (B*G, Ns, C) where C = M * D_h
        v = self.v_proj(sampled_features)  # (B*G, Ns, C)

        # 6. 重塑 K, V 为多头形式
        # 
        # 步骤6a: 将 C 维度拆分为多头
        # (B*G, Ns, C) -> (B*G, Ns, M, D_h)
        k = k.reshape(B * G, Ns, M, D_h)  # (B*G, Ns, M, D_h)
        v = v.reshape(B * G, Ns, M, D_h)  # (B*G, Ns, M, D_h)
        
        # 步骤6b: 重新组织为 batch 维度
        # (B*G, Ns, M, D_h) -> (B, G, Ns, M, D_h)
        k = k.reshape(B, G, Ns, M, D_h)  # (B, G, Ns, M, D_h)
        v = v.reshape(B, G, Ns, M, D_h)  # (B, G, Ns, M, D_h)
        
        # 步骤6c: 每个组负责 M//G 个头
        # 我们需要从每个组中选择对应的头
        # 组0 负责头 [0, M//G)
        # 组1 负责头 [M//G, 2*M//G)
        # ...
        # 
        # (B, G, Ns, M, D_h) -> (B, G, Ns, M//G, D_h)
        # 对每个组选择对应的头切片
        heads_per_group = M // G
        k_grouped = []
        v_grouped = []
        
        for g in range(G):
            start_head = g * heads_per_group
            end_head = start_head + heads_per_group
            k_grouped.append(k[:, g:g+1, :, start_head:end_head, :])  # (B, 1, Ns, M//G, D_h)
            v_grouped.append(v[:, g:g+1, :, start_head:end_head, :])  # (B, 1, Ns, M//G, D_h)
        
        # (B, G, Ns, M//G, D_h) -> (B, M, Ns, D_h)
        k = torch.cat(k_grouped, dim=1)  # (B, G, Ns, M//G, D_h)
        v = torch.cat(v_grouped, dim=1)  # (B, G, Ns, M//G, D_h)
        
        # 重塑: (B, G, Ns, M//G, D_h) -> (B, G, M//G, Ns, D_h) -> (B, M, Ns, D_h)
        k = k.permute(0, 1, 3, 2, 4).reshape(B, M, Ns, D_h)  # (B, M, Ns, D_h)
        v = v.permute(0, 1, 3, 2, 4).reshape(B, M, Ns, D_h)  # (B, M, Ns, D_h)
        
        # 最终: k, v 都是 (B, M, Ns, D_h)

        # 7. 计算 Attention
        # Q: (B, M, N, D_h)  每个token都有query
        # K: (B, M, Ns, D_h) 只有Ns个采样点有key
        # (B, M, N, D_h) @ (B, M, D_h, Ns) -> (B, M, N, Ns)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, M, N, Ns)

        # 8. 添加 RPB (相对位置偏置)
        if self.use_rpb:
            # (B, Ns, G, 2) -> (B, M, 1, Ns)
            rpb = self._sample_rpb(deformed_points)  # (B, M, 1, Ns)
            attn = attn + rpb  # (B, M, N, Ns) + (B, M, 1, Ns) -> (B, M, N, Ns)
        
        attn = attn.softmax(dim=-1)  # (B, M, N, Ns) 对采样点维度softmax

        # 9. 计算输出
        # (B, M, N, Ns) @ (B, M, Ns, D_h) -> (B, M, N, D_h)
        out = (attn @ v)  # (B, M, N, D_h)
        # (B, M, N, D_h) -> (B, N, M, D_h) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)  # (B, N, C)
        
        return out  # (B, N, C)

#
# 步骤 3: 构建完整的 Transformer Block
#

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, grid_size, offset_groups,
                 mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.offset_groups = offset_groups
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DeformableAttention(
            dim, num_heads, grid_size, offset_groups
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#
# 步骤 4: 组装成 DATA 论文  中的 DAT-base 模型
#

class DAT_base_for_sim(nn.Module):
    """
    DAT-base 模型
    配置来自: Zhao et al. 2025 (DATA), Table I 
    Offset-Groups (G) 来自: Xia et al. 2023 (DAT++), Table 1 [cite: 1148] (G=1, 2, 4, 8)
    Grid-Size (Window Size) = 7 
    
    模型结构:
        Stage1: 128-dim,  4-heads, 1-block,  G=1
        Stage2: 256-dim,  8-heads, 1-block,  G=2
        Stage3: 512-dim, 16-heads, 9-blocks, G=4
        Stage4: 1024-dim, 32-heads, 1-block,  G=8
    
    输入输出:
        输入: (B, 3, 224, 224) RGB图像
        输出: (B, num_classes) 分类logits
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[128, 256, 512, 1024],
                 num_heads=[4, 8, 16, 32],
                 num_blocks=[1, 1, 9, 1],
                 offset_groups=[1, 2, 4, 8], # DAT++ G (T, S, B)
                 grid_size=7,
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.depths = num_blocks
        self.num_stages = len(self.depths)
        
        # DropPath随着深度递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        cur = 0

        # 1. Patch 嵌入: (B, 3, 224, 224) -> (B, 3136, 128) with hw=(56, 56)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )

        # 2. 构建 4 个 Stage (每个Stage包含多个TransformerBlock，后跟Downsample)
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            # 构建当前Stage的所有Block
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    grid_size=grid_size,
                    offset_groups=offset_groups[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]
                )
                for j in range(num_blocks[i])
            ])
            
            # 将Block列表作为一个整体添加到Stage
            self.stages.append(blocks)
            
            # 在Stage之间添加Downsample (最后一个Stage除外)
            if i < self.num_stages - 1:
                self.stages.append(
                    Downsample(embed_dims[i], embed_dims[i+1])
                )
            
            cur += num_blocks[i]

        # 3. 分类头
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (B, 3, H, W) 输入图像，默认 H=W=224
        
        返回:
            x: (B, num_classes) 分类logits
        
        详细流程:
            PatchEmbed:  (B, 3, 224, 224) -> (B, 3136, 128), hw=(56, 56)
            Stage1:      (B, 3136, 128) -> (B, 3136, 128)
            Downsample1: (B, 3136, 128) -> (B, 784, 256), hw=(28, 28)
            Stage2:      (B, 784, 256) -> (B, 784, 256)
            Downsample2: (B, 784, 256) -> (B, 196, 512), hw=(14, 14)
            Stage3:      (B, 196, 512) -> (B, 196, 512) (9个Block)
            Downsample3: (B, 196, 512) -> (B, 49, 1024), hw=(7, 7)
            Stage4:      (B, 49, 1024) -> (B, 49, 1024)
            GAP + Head:  (B, 49, 1024) -> (B, 1024) -> (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch嵌入
        x, hw_shape = self.patch_embed(x)  # (B, N, C), (H, W)
        
        # 遍历所有Stage
        # stages的结构: [blocks, downsample, blocks, downsample, blocks, downsample, blocks]
        # 索引:          [0,      1,          2,      3,          4,      5,          6]
        stage_idx = 0
        for i in range(0, len(self.stages), 2):  # i = 0, 2, 4, 6
            # 当前Stage的所有Block
            blocks = self.stages[i]
            
            # 逐个执行Block (不能用nn.Sequential，因为需要传递hw_shape)
            for block in blocks:
                x = block(x, hw_shape)  # (B, N, C) -> (B, N, C)
            
            # 如果不是最后一个Stage，执行Downsample
            if i + 1 < len(self.stages):
                downsample = self.stages[i + 1]
                x, hw_shape = downsample(x, hw_shape)  # (B, N, C) -> (B, N/4, C*2), (H/2, W/2)
            
            stage_idx += 1

        # 分类头
        x = self.norm(x)  # (B, N, C)
        x = x.mean(dim=1)  # Global Average Pooling: (B, N, C) -> (B, C)
        x = self.head(x)  # (B, C) -> (B, num_classes)
        
        return x