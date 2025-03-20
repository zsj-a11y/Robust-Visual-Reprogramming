# Robust Visual Reprogramming

The code for Paper "Endowing Visual Reprogramming with Adversarial Robustness"



Examples:

Adversarial visual reprogramming on CIFAR-10 using adversarial pre-trained ResNet-18:

```bash
python main.py --dataset CIFAR10 --network Salman2020Do_R18 --attack pgd --vp full --mapping_method fc --lr_vp 1e-3 --lr_lm 1e-3 --wd 1e-3 --seed 0 --gpu 5 --bs 256 --port 1205
```

