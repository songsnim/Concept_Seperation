import torch
import torch.nn as nn



class Mapper(nn.Module):
    def __init__(self,args):
        super(Mapper, self).__init__()
        
        self.reduce_layer = nn.Sequential(nn.Linear(64*4*4, args['reduced_dim']), nn.ReLU(),nn.BatchNorm1d(args['reduced_dim']))
        
        self.cont_layer_P = nn.Sequential(nn.Linear(args['reduced_dim'], args['cont_dim_P']), nn.ReLU(),nn.BatchNorm1d(args['cont_dim_P']))
        self.cont_layer_G = nn.Sequential(nn.Linear(args['reduced_dim'], args['cont_dim_G']), nn.ReLU(),nn.BatchNorm1d(args['cont_dim_G']))
        self.latent_layer = nn.Sequential(nn.Linear(args['reduced_dim'], args['latent_dim']), nn.ReLU(),nn.BatchNorm1d(args['latent_dim']))
        
    def forward(self, encoded):
        encoded = encoded.view(-1,64*4*4)
        reduced = self.reduce_layer(encoded)
        cont_code_P = self.cont_layer_P(reduced)
        cont_code_G = self.cont_layer_G(reduced)
        latent = self.latent_layer(reduced)
        return cont_code_P, cont_code_G, latent
        

class Predictor(nn.Module):
    def __init__(self,args):
        super(Predictor, self).__init__()
            
        self.input_dim = args['cont_dim_P']
        self.predictor = nn.Sequential(nn.Linear(self.input_dim, 128),nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(128, 64), nn.ReLU(),nn.Dropout(0.2),
                                        nn.Linear(64,2)) # predict class (1 or 7)
    def forward(self, cont_code_P):
        predicted = self.predictor(cont_code_P)
        return predicted

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        input_dim = args['reduced_dim'] + args['n_classes'] # 32 + 2 + 2 + 1
        
        self.init_size = args['img_size'] // 4
        # conv에 넣을 수 있도록 dim_adjust
        self.fc = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size * self.init_size ))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256), nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,3, stride=1, padding=1), nn.BatchNorm2d(256,0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,128,3, stride=1, padding=1), nn.BatchNorm2d(128,0.8), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3, stride=1, padding=1), nn.BatchNorm2d(64,0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,32,3, stride=1, padding=1), nn.BatchNorm2d(32,0.8), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32,args['channels'], 3, stride=1, padding=1), 
            nn.Tanh()
        )
        
    def forward(self, labels, cont_code_P, cont_code_G, latent):
        gen_input = torch.cat((labels, cont_code_P, cont_code_G, latent), dim=-1) # cat => 32+2+2+1 = 37 \
        # print(gen_input.shape, args['reduced_dim'])
        #  mat1 and mat2 shapes cannot be multiplied (128x37 and 36x8192)
        out = self.fc(gen_input)    
        out = out.view(out.shape[0], 256, self.init_size, self.init_size) # block화
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
        self.cont_P_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, 
                                                    args['cont_dim_P'])) # cont_code_P 예측
        self.cont_G_layer = nn.Sequential(nn.Linear(128*downsample_size*downsample_size, 
                                                    args['cont_dim_G'])) # cont_code_G 예측
        
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        reality = self.adv_layer(out)
        pred_label = self.disc_layer(out)
        pred_cont_P = self.cont_P_layer(out)
        pred_cont_G = self.cont_G_layer(out)
        
        return reality, pred_label, pred_cont_P, pred_cont_G