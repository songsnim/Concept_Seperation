import torch
import torch.nn as nn



class Mapper(nn.Module):
    def __init__(self,args):
        super(Mapper, self).__init__()
        
        self.reduce_layer = nn.Sequential(nn.Linear(64*4*4, args['reduced_dim'],bias=False), 
                                          nn.ReLU(),nn.BatchNorm1d(args['reduced_dim']))
        
        self.code_P_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['code_P_dim'],bias=False), 
                                          nn.ReLU(),nn.BatchNorm1d(args['code_P_dim']))
        self.code_G_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['code_G_dim'],bias=False), 
                                          nn.ReLU(),nn.BatchNorm1d(args['code_G_dim']))
        self.latent_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['latent_dim'],bias=False), 
                                          nn.ReLU(),nn.BatchNorm1d(args['latent_dim']))
        
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
        input_dim = args['reduced_dim']  # 32 + 2 + 1 = 35
        
        self.init_size = args['img_size'] // 4
        # conv에 넣을 수 있도록 dim_adjust
        self.fc = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size * self.init_size ))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3, stride=1, padding=1), nn.BatchNorm2d(128,0.8), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3, stride=1, padding=1), nn.BatchNorm2d(64,0.8), nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(64,args['channels'], 3, stride=1, padding=1), 
            nn.Tanh()
        )
        
    def forward(self, code_P, code_G, latent):
        gen_input = torch.cat((code_P, code_G, latent), dim=-1) # cat => 32+2+1 = 35 \
        out = self.fc(gen_input)    
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) # block화
        img = self.conv_blocks(out)
        
        return img

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
        pred_code_P = self.code_P_layer(out)
        pred_code_G = self.code_G_layer(out)
        
        return reality, pred_code_P, pred_code_G