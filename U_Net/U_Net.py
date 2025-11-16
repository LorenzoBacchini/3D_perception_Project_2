import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, n_filters=64, dropout_prob=0.3, max_pooling=True):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_ch, n_filters, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same')
        self.activation = nn.ReLU()
        self.max_pooling = max_pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if self.max_pooling else None
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.activation(out)
        out = self.batch_norm(out)
        if self.dropout_prob > 0:
            out = self.dropout(out)

        skip_connection = out.clone()
        if self.max_pooling:
            next_layer = self.pool(out)
        else:
            next_layer = out

        return next_layer, skip_connection
    
class upsampling_block(nn.Module):
    def __init__(self, in_ch, skip_ch, n_filters=64):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=(2, 2), padding=0)
        self.conv_1 = nn.Conv2d(in_ch//2 + skip_ch, n_filters, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same')
        self.activation = nn.ReLU()


    def forward(self, x, skip):
        conv = self.upsample(x)
        conv = torch.cat([conv, skip], dim=1)
        conv = self.conv_1(conv)
        conv = self.activation(conv)
        conv = self.conv_2(conv)
        conv = self.activation(conv)

        return conv

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_filters=64, n_classes=19):
        super().__init__()
        self.conv_layer_1 = conv_block(in_ch, n_filters, dropout_prob=0, max_pooling=True)
        self.conv_layer_2 = conv_block(n_filters, n_filters*2, dropout_prob=0, max_pooling=True)
        self.conv_layer_3 = conv_block(n_filters*2, n_filters*4, dropout_prob=0, max_pooling=True)
        self.conv_layer_4 = conv_block(n_filters*4, n_filters*8, dropout_prob=0.3, max_pooling=True)
        self.conv_layer_5 = conv_block(n_filters*8, n_filters*16, dropout_prob=0.3, max_pooling=False)

        self.upsample_layer_1 = upsampling_block(n_filters*16, n_filters*8, n_filters*8)
        self.upsample_layer_2 = upsampling_block(n_filters*8, n_filters*4, n_filters*4)
        self.upsample_layer_3 = upsampling_block(n_filters*4, n_filters*2, n_filters*2)
        self.upsample_layer_4 = upsampling_block(n_filters*2, n_filters, n_filters)

        self.last_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_classes, kernel_size=1, padding='same')
        )

    def forward(self, x):
        conv_1_next, conv_1_skip = self.conv_layer_1(x)
        conv_2_next, conv_2_skip = self.conv_layer_2(conv_1_next)
        conv_3_next, conv_3_skip = self.conv_layer_3(conv_2_next)
        conv_4_next, conv_4_skip = self.conv_layer_4(conv_3_next)
        conv_5_next, conv_5_skip = self.conv_layer_5(conv_4_next)

        out = self.upsample_layer_1(conv_5_next, conv_4_skip)
        out = self.upsample_layer_2(out, conv_3_skip)
        out = self.upsample_layer_3(out, conv_2_skip)
        out = self.upsample_layer_4(out, conv_1_skip)
        out = self.last_conv(out)

        return out