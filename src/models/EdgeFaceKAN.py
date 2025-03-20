import torch
from torch import nn
from timm.layers import trunc_normal_
from ..layers.custom_layers import LayerNorm, PositionalEncodingFourier
from ..layers.sdta_encoder import SDTAEncoder
from ..layers.conv_encoder import ConvEncoder
from ..layers.LoRaLin import LoRaLin
from ..layers.KANLinear import KANLinear

class EdgeFaceKAN(nn.Module):
    def __init__(self, in_chans=3, num_features=512, rank_ratio = 0.6,
                 depths=[3, 3, 9, 3], dims=[32, 64, 100, 192],
                 global_block=[0, 1, 1, 1], global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[3, 5, 7, 9], heads=[4, 4, 4, 4], use_pos_embd_xca=[False, True, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 2, 3, 4], grid_size=5, spline_order=3,
                 base_activation=nn.SiLU(), neuron_fun=None, noise_type=None):
        super().__init__()
        if rank_ratio == 0.6:
            dims=[32, 64, 100, 192]
        elif rank_ratio == 0.5:
            dims=[48, 96, 160, 304]
        for g in global_block_type:
            assert g in ['None', 'SDTA']
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA':
                        stage_blocks.append(SDTAEncoder(dim=dims[i], rank_ratio=rank_ratio,drop_path=dp_rates[cur + j],
                                                        expan_ratio=expan_ratio, scales=d2_scales[i],
                                                        use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[i], rank_ratio=rank_ratio,drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        print(f"grid_size in EdgeFaceKAN: {grid_size}")
        self.head = KANLinear(dims[-1], num_features,
            num=grid_size,
            k=spline_order,
            base_activation=base_activation,
            neuron_fun=neuron_fun,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
