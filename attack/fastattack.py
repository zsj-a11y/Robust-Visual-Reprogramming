from attack.libfastattack.FastPGD import FastPGD
from attack.libfastattack.FastTRADES import FastTRADES


def attack_loader(net, attack, eps, steps, alpha):

    
    if attack == "pgd":
        return FastPGD(model=net, eps=eps, alpha=alpha, steps=steps, random_start=True)
    
    elif attack == "trades":
        return FastTRADES(model=net, eps=eps,
                                alpha=alpha, steps=steps, random_start=True)
    elif attack == "nt":
        from torchattacks.attack import Attack
        class NT(Attack):
            def __init__(self, model):
                super().__init__("NT", model)
            def forward(self, x, y):
                return x
        return NT(net)

