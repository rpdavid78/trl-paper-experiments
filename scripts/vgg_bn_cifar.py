# vgg_bn_cifar.py
# VGG-11-BN adaptado para CIFAR, projetado como "drop-in" para o pipeline TRL
# existente (cifar100_all_methods_iclr.py).
#
# DECISOES DE DESIGN (fixadas ANTES de ver qualquer numero, para nao cair em
# garden-of-forking-paths):
#   1) A head se chama EXATAMENTE self.linear  -> casa com:
#        - build_trl_prior_from_laplace:  ("linear." in name or "fc." in name)
#        - laplace-torch subset_of_weights="last_layer"
#   2) Uma unica nn.Linear(512, num_classes) apos pooling (sem as 2 FC de 4096
#      do VGG classico). Mantem a topologia "backbone conv + 1 head linear"
#      identica a ResNetCIFAR, para a divisao de prior conv-vs-head ser justa
#      em vez de confundida por uma cabeca gigante.
#   3) Dropout governado por use_dropout/p_drop, inserido apos cada bloco conv
#      (apos o ReLU pos-BN) e antes da head (drop_head), espelhando os pontos
#      da BasicBlock/ResNetCIFAR. Assim o mc_dropout_predict existente roda sem
#      modificacao e a comparacao TRL vs MC-Dropout fica apples-to-apples.
#   4) VGG-11 (config "A" com BN). E o VGG mais leve; ~9.8M params com head
#      CIFAR-100, mesma ordem de grandeza da ResNet-18, logo o eigsh sobre HVP
#      roda no mesmo regime de custo da Tab. 8.
#
# A interface (assinatura __init__ e forward) e identica a ResNetCIFAR:
#   ResNetCIFAR(num_classes=100, use_dropout=False, p_drop=0.0)
# para que load_or_train_map / load_or_train_mcdo possam troca-la 1:1.

import torch
import torch.nn as nn

# Config "A" do VGG (VGG-11). 'M' = maxpool. Numeros = canais de saida do conv.
_VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGGCIFAR(nn.Module):
    """VGG-11-BN para CIFAR. Mesma interface de ResNetCIFAR."""

    def __init__(self, num_classes: int = 100, use_dropout: bool = False, p_drop: float = 0.0):
        super().__init__()
        self.use_dropout = use_dropout
        self.p_drop = p_drop

        self.features = self._make_layers(_VGG11_CFG, use_dropout=use_dropout, p_drop=p_drop)

        # Apos 5 maxpools, 32x32 -> 1x1 ja no VGG-11, mas usamos AdaptiveAvgPool
        # para robustez (garante 1x1 independente de quirks de borda).
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_head = nn.Dropout(p_drop) if (use_dropout and p_drop > 0) else nn.Identity()

        # IMPORTANTE: nome 'linear' (nao 'classifier') para casar com o prior
        # split e com o last-layer Laplace.
        self.linear = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layers(self, cfg, use_dropout: bool, p_drop: float):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                # Dropout apos o bloco conv+BN+ReLU, espelhando self.drop da
                # BasicBlock. So entra quando use_dropout (modelo do MC-Dropout).
                if use_dropout and p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out).flatten(1)
        out = self.drop_head(out)
        return self.linear(out)


if __name__ == "__main__":
    # Smoke test local: contagem de params + checagem de nomes + forward.
    for nc in (10, 100):
        net = VGGCIFAR(num_classes=nc, use_dropout=False)
        n = sum(p.numel() for p in net.parameters())
        head_names = [name for name, _ in net.named_parameters() if "linear." in name]
        x = torch.randn(4, 3, 32, 32)
        y = net(x)
        print(f"num_classes={nc}: params={n/1e6:.2f}M  out={tuple(y.shape)}  head_params={head_names}")

    # Versao MC-Dropout: confere que ha modulos Dropout para o enable_dropout_only.
    netd = VGGCIFAR(num_classes=100, use_dropout=True, p_drop=0.2)
    n_drop = sum(1 for m in netd.modules() if isinstance(m, nn.Dropout))
    print(f"MC-Dropout variant: {n_drop} Dropout modules (esperado > 0)")
