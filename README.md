# Robust Visual Reprogramming

The code for Paper "Endowing Visual Reprogramming with Adversarial Robustness"

Abstract:
Visual reprogramming (VR) leverages well-developed pre-trained models (e.g., a pre-trained classifier on ImageNet) to tackle target tasks (e.g., a traffic sign recognition task), without the need for training from scratch. Despite the effectiveness of previous VR methods, all of them did not consider the adversarial robustness of reprogrammed models against adversarial attacks, which could lead to unpredictable problems in safety-crucial target tasks. In this paper, we empirically find that reprogramming pre-trained models with adversarial robustness and incorporating adversarial samples from the target task during reprogramming can both improve the adversarial robustness of reprogrammed models. Furthermore, we propose a theoretically guaranteed adversarial robustness risk upper bound for VR, which validates our empirical findings and could provide a theoretical foundation for future research. Extensive experiments demonstrate that by adopting the strategies revealed in our empirical findings, the adversarial robustness of reprogrammed models can be enhanced.



Examples:

Adversarial visual reprogramming on CIFAR-10 using adversarial pre-trained ResNet-18:

```bash
python main.py --dataset CIFAR10 --network Salman2020Do_R18 --attack pgd --vp full --mapping_method fc --lr_vp 1e-3 --lr_lm 1e-3 --wd 1e-3 --seed 0 --gpu 5 --bs 256 --port 1205
```

