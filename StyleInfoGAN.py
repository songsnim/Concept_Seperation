import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_linear(linear):
    nn.init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv):
    nn.init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

def calc_mean_std(tensor, eps=1e-5): 
    n, c, h, w = tensor.shape
    tensor_var = torch.var(tensor.reshape(n, c, -1), axis=2) + eps
    tensor_std = torch.sqrt(tensor_var).reshape(n, c, 1, 1)
    tensor_mean = torch.mean(tensor.reshape(n, c, -1), axis=2).reshape(n, c, 1, 1) 
    
    return tensor_mean, tensor_std

def adaptive_instance_normalization(content, style):
    assert (content.size()[:2] == style.size()[:2])
    size = content.size()
    style_mean, style_std = calc_mean_std(style)
    content_mean, content_std = calc_mean_std(content)
    normalized_tensor = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalized_tensor * style_std.expand(size) + style_mean.expand(size)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image):
        return image + self.weight.to(image.device) * torch.randn(self.weight.size()).to(image.device)



class Mapper(nn.Module):
    def __init__(self,args):
        super(Mapper, self).__init__()
        
        self.reduce_layer = nn.Sequential(nn.Linear(64*4*4, args['reduced_dim']), nn.ReLU(),nn.BatchNorm1d(args['reduced_dim']))
        
        self.code_P_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['code_P_dim']), nn.ReLU(),nn.BatchNorm1d(args['code_P_dim']))
        self.code_G_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['code_G_dim']), nn.ReLU(),nn.BatchNorm1d(args['code_G_dim']))
        self.latent_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['latent_dim']), nn.ReLU(),nn.BatchNorm1d(args['latent_dim']))
        
    def forward(self, encoded):
        encoded = encoded.view(-1,64*4*4)
        reduced = self.reduce_layer(encoded)
        code_P = self.code_P_layer(reduced)
        code_G = self.code_G_layer(reduced)
        latent = self.latent_layer(reduced)
        return code_P, code_G, latent
        
class Predictor(nn.Module):
    def __init__(self,args):
        super(Predictor, self).__init__()
            
        self.input_dim = args['code_P_dim']
        self.predictor = nn.Sequential(nn.Linear(self.input_dim, 128),nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(128, 64), nn.ReLU(),nn.Dropout(0.2),
                                        nn.Linear(64,2)) # predict class (1 or 7)
        # self.simple_predictor = nn.Linear(self.input_dim, 2)
        
    def forward(self, code_P):
        predicted = self.predictor(code_P)
        return predicted

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        input_dim = args['latent_dim'] + args['n_classes'] # 32 + 2 + 2 + 1
        
        self.init_size = args['img_size'] // 4
        self.fc = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size * self.init_size ))# conv에 넣을 수 있도록 dim_adjust
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.leaky_relu_init = nn.LeakyReLU(0.2)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        
        self.linear_init = nn.Linear(3, 256*self.init_size*self.init_size*4) ; self.noise_injector_init = NoiseInjection(256)
        self.linear_1 = nn.Linear(3,128*self.init_size*self.init_size*4**2)  ; self.noise_injector_1 = NoiseInjection(128)
        self.linear_2 = nn.Linear(3,64*self.init_size*self.init_size*4**2)   ; self.noise_injector_2 = NoiseInjection(64)
        
        self.initial_layer = nn.Sequential(
            nn.BatchNorm2d(256), nn.Upsample(scale_factor=2)
            )
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(256,128,3, stride=1, padding=1),
            nn.Conv2d(128,128,3, stride=1, padding=1),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128,64,3, stride=1, padding=1),
            nn.Conv2d(64, 64,3 ,stride=1, padding=1),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64,args['channels'], 3, stride=1, padding=1), 
            nn.Tanh()
        )
        
        
    def forward(self, labels, code_P, code_G, latent ):
        gen_input = torch.cat((labels,latent), dim=-1) # cat => 2 + 32 = 34 \
        gen_input = self.fc(gen_input)
        gen_input = gen_input.view(gen_input.shape[0], 256, self.init_size, self.init_size) # block화
        
        code = torch.cat((code_P, code_G), dim=-1)
        
        out_init = self.initial_layer(gen_input)
        out_init = self.noise_injector_init(out_init)
        out_init = self.leaky_relu_init(out_init)
        code_init_expanded = self.linear_init(code)
        code_init_blocked = code_init_expanded.view(out_init.size())
        out_init = adaptive_instance_normalization(out_init, code_init_blocked)
        

        out_1 = self.conv_block_1(out_init)
        out_1 = self.upsample(out_1)
        out_1 = self.noise_injector_1(out_1)
        out_1 = self.leaky_relu_1(out_1)
        code_1_expanded = self.linear_1(code)
        code_1_blocked = code_1_expanded.view(out_1.size())
        out_1 = adaptive_instance_normalization(out_1, code_1_blocked)
        
        out_2 = self.conv_block_2(out_1)
        out_2 = self.noise_injector_2(out_2)
        out_2 = self.leaky_relu_2(out_2)
        code_2_expanded = self.linear_2(code)
        code_2_blocked = code_2_expanded.view(out_2.size())
        out_2 = adaptive_instance_normalization(out_2, code_2_blocked)
        
        out_3 = self.conv_block_3(out_2)
        
        return out_3

class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        # batchnorm을 넣을지 말지에 따라서 discriminator_block의 모듈 수가 달라진다
        # 달라지면 nn.Sequential에 추가할 때 if로 나눠서 해야하나? 싶지만
        # *를 사용하면 block안에 모듈이 몇개든 그냥 싹다 넣어주는 역할을 한다.
        self.conv_blocks = nn.Sequential(
            *discriminator_block(args['channels'], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )
        
        downsample_size = args['img_size'] // (2**4) #stride 2인 block이 4개 있었으니까 4번 downsampled 
        
        self.adv_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, 1)) # real or fake 예측 
        self.disc_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, args['n_classes']),
                                       nn.Softmax()) # class 예측
        self.code_P_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, 
                                                    args['code_P_dim'])) # code_P 예측
        self.code_G_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, 
                                                    args['code_G_dim'])) # code_G 예측
        
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
    
        reality = self.adv_layer(out)
        pred_label = self.disc_layer(out)
        pred_code_P = self.code_P_layer(out)
        pred_code_G = self.code_G_layer(out)
        
        return reality, pred_label, pred_code_P, pred_code_G