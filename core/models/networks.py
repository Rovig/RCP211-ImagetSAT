#!/usr/bin/env python3
"""
Architectures Pix2Pix partagées par l'entraînement et l'inférence
"""

import torch
import torch.nn as nn

class ImprovedGenerator(nn.Module):
    """Générateur amélioré avec plus de couches et skip connections."""
    
    def __init__(self, input_channels=3, output_channels=3, ngf=64, image_size=256):
        super(ImprovedGenerator, self).__init__()
        self.image_size = image_size
        
        if image_size >= 512:
            # Pour 512x512 et plus
            self.down1 = self._make_layer(input_channels, ngf, normalize=False)  # 512->256
            self.down2 = self._make_layer(ngf, ngf*2)                           # 256->128
            self.down3 = self._make_layer(ngf*2, ngf*4)                        # 128->64
            self.down4 = self._make_layer(ngf*4, ngf*8)                        # 64->32
            self.down5 = self._make_layer(ngf*8, ngf*8)                        # 32->16
            self.down6 = self._make_layer(ngf*8, ngf*8)                        # 16->8
            self.down7 = self._make_layer(ngf*8, ngf*8)                        # 8->4
            self.down8 = self._make_layer(ngf*8, ngf*8, normalize=False)       # 4->2
            self.down9 = self._make_layer(ngf*8, ngf*8, normalize=False)       # 2->1
            
            self.up1 = self._make_uplayer(ngf*8, ngf*8, dropout=True)          # 1->2
            self.up2 = self._make_uplayer(ngf*16, ngf*8, dropout=True)         # 2->4
            self.up3 = self._make_uplayer(ngf*16, ngf*8, dropout=True)         # 4->8
            self.up4 = self._make_uplayer(ngf*16, ngf*8, dropout=True)         # 8->16
            self.up5 = self._make_uplayer(ngf*16, ngf*8)                       # 16->32
            self.up6 = self._make_uplayer(ngf*16, ngf*4)                       # 32->64
            self.up7 = self._make_uplayer(ngf*8, ngf*2)                        # 64->128
            self.up8 = self._make_uplayer(ngf*4, ngf)                          # 128->256
            self.up9 = nn.Sequential(                                          # 256->512
                nn.ConvTranspose2d(ngf*2, output_channels, 4, 2, 1),
                nn.Tanh()
            )
            self.use_9_layers = True
        else:
            # Pour 256x256
            self.down1 = self._make_layer(input_channels, ngf, normalize=False)  # 256->128
            self.down2 = self._make_layer(ngf, ngf*2)                           # 128->64
            self.down3 = self._make_layer(ngf*2, ngf*4)                        # 64->32
            self.down4 = self._make_layer(ngf*4, ngf*8)                        # 32->16
            self.down5 = self._make_layer(ngf*8, ngf*8)                        # 16->8
            self.down6 = self._make_layer(ngf*8, ngf*8)                        # 8->4
            self.down7 = self._make_layer(ngf*8, ngf*8)                        # 4->2
            self.down8 = self._make_layer(ngf*8, ngf*8, normalize=False)       # 2->1
            
            self.up1 = self._make_uplayer(ngf*8, ngf*8, dropout=True)          # 1->2
            self.up2 = self._make_uplayer(ngf*16, ngf*8, dropout=True)         # 2->4
            self.up3 = self._make_uplayer(ngf*16, ngf*8, dropout=True)         # 4->8
            self.up4 = self._make_uplayer(ngf*16, ngf*8)                       # 8->16
            self.up5 = self._make_uplayer(ngf*16, ngf*4)                       # 16->32
            self.up6 = self._make_uplayer(ngf*8, ngf*2)                        # 32->64
            self.up7 = self._make_uplayer(ngf*4, ngf)                          # 64->128
            self.up8 = nn.Sequential(                                          # 128->256
                nn.ConvTranspose2d(ngf*2, output_channels, 4, 2, 1),
                nn.Tanh()
            )
            self.use_9_layers = False
    
    def _make_layer(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _make_uplayer(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_9_layers:
            # Encoder pour 512x512
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)
            d9 = self.down9(d8)
            
            # Decoder avec skip connections
            u1 = self.up1(d9)
            u2 = self.up2(torch.cat([u1, d8], 1))
            u3 = self.up3(torch.cat([u2, d7], 1))
            u4 = self.up4(torch.cat([u3, d6], 1))
            u5 = self.up5(torch.cat([u4, d5], 1))
            u6 = self.up6(torch.cat([u5, d4], 1))
            u7 = self.up7(torch.cat([u6, d3], 1))
            u8 = self.up8(torch.cat([u7, d2], 1))
            u9 = self.up9(torch.cat([u8, d1], 1))
            return u9
        else:
            # Encoder pour 256x256
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)
            
            # Decoder avec skip connections
            u1 = self.up1(d8)
            u2 = self.up2(torch.cat([u1, d7], 1))
            u3 = self.up3(torch.cat([u2, d6], 1))
            u4 = self.up4(torch.cat([u3, d5], 1))
            u5 = self.up5(torch.cat([u4, d4], 1))
            u6 = self.up6(torch.cat([u5, d3], 1))
            u7 = self.up7(torch.cat([u6, d2], 1))
            u8 = self.up8(torch.cat([u7, d1], 1))
            return u8


class ImprovedDiscriminator(nn.Module):
    """Discriminateur amélioré avec adaptation à la taille d'image."""
    
    def __init__(self, input_channels=6, ndf=64, image_size=256):
        super(ImprovedDiscriminator, self).__init__()
        
        if image_size >= 512:
            # Pour 512x512: 5 couches de downsampling
            self.model = nn.Sequential(
                # 512 -> 256
                nn.Conv2d(input_channels, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 256 -> 128
                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 128 -> 64
                nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 64 -> 32
                nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 32 -> 16
                nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 16 -> sortie
                nn.Conv2d(ndf*8, 1, 4, 1, 1)
            )
        else:
            # Pour 256x256: 4 couches de downsampling  
            self.model = nn.Sequential(
                # 256 -> 128
                nn.Conv2d(input_channels, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 128 -> 64
                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 64 -> 32
                nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 32 -> 16
                nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 16 -> sortie (adaptation pour 256x256)
                nn.Conv2d(ndf*8, 1, 4, 1, 1)
            )
    
    def forward(self, input, target=None):
        # target=None: input est déjà la paire (carte, image) concaténée
        x = input if target is None else torch.cat([input, target], 1)
        return self.model(x)


def generator_from_state_dict(state_dict):
    """Construit le générateur dont l'architecture correspond au state_dict."""
    # Seule l'architecture 512 possède une 9e couche de downsampling
    image_size = 512 if 'down9.0.weight' in state_dict else 256
    generator = ImprovedGenerator(image_size=image_size)
    generator.load_state_dict(state_dict)
    return generator


def discriminator_from_state_dict(state_dict):
    """Construit le discriminateur dont l'architecture correspond au state_dict."""
    # Seule l'architecture 512 possède une 5e couche de downsampling (model.14)
    image_size = 512 if 'model.14.weight' in state_dict else 256
    discriminator = ImprovedDiscriminator(image_size=image_size)
    discriminator.load_state_dict(state_dict)
    return discriminator
