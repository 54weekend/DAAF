from torch import nn
from torch.hub import load_state_dict_from_url

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.out_channels = out_planes

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        if inverted_residual_setting is None:
            inverted_residual_setting_darks_3 = [
                [1, 32, 1, 1],
                [6, 64, 2, 2],
                [6, 128, 3, 2],
            ]
            inverted_residual_setting_darks_4 = [
                [6, 128, 4, 2],
                [6, 256, 3, 1],
            ]
            inverted_residual_setting_darks_5 = [
                [6, 256, 3, 2],
                [6, 256, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features_derk_3 = [ConvBNReLU(3, input_channel, stride=2)]
        features_derk_4 = []
        features_derk_5 = []
        for t, c, n, s in inverted_residual_setting_darks_3:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_derk_3.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting_darks_4:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_derk_4.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting_darks_5:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_derk_5.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.dark3 = nn.Sequential(*features_derk_3)
        self.dark4 = nn.Sequential(*features_derk_4)
        self.dark5 = nn.Sequential(*features_derk_5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

def mobilenet_v2():
    model = MobileNetV2()

    return model

if __name__ == "__main__":
    net = mobilenet_v2()
    for i, layer in enumerate(net.features):
        print(i, layer)
