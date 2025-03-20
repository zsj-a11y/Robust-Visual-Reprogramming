import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter



from model.input_transformation import InstancewiseVisualPrompt, InputNormalize, ExpansiveVisualPrompt, FullyVisualPrompt
from model.output_mapping import LabelMappingBase, generate_label_mapping_by_frequency, linear
from model.pipline import Pipeline
from utils.data import prepare_data, IMAGENETNORMALIZE
from attack.fastattack import attack_loader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def print_configuration(args, rank):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys():
            print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')
        

def get_image_size(args):
    if args.source_task == "cifar100":
        imgsize = 32
    elif args.source_task == 'imagenet':
        imgsize = 224
    else:
        imgsize = 32
    return imgsize

def get_dataloader(args):
    loaders, class_names, train_sampler = prepare_data(args.dataset, args.data_path, batch_size=args.bs)
    return loaders, class_names, train_sampler

def get_model(args, loaders, class_names):
    imgsize = get_image_size(args)
    
    if args.source_task == 'cifar100':
        source_out = 100
    elif args.source_task == 'imagenet':
        source_out = 1000
    # Network
    if args.network == "Salman2020Do_R18": 
        from robustbench import load_model
        network = load_model(model_name='Salman2020Do_R18', dataset=args.source_task, threat_model='Linf')
    elif args.network == "Salman2020Do_R50": 
        from robustbench import load_model 
        network = load_model(model_name='Salman2020Do_R50', dataset=args.source_task, threat_model='Linf')
    elif args.network == "Addepalli2022Efficient_RN18":
        from robustbench import load_model 
        network = load_model(model_name='Addepalli2022Efficient_RN18', dataset=args.source_task, threat_model='Linf')
        network = InputNormalize(network, [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    elif args.network == "Debenedetti2022Light_XCiT-S12":
        from robustbench import load_model 
        network = load_model(model_name='Debenedetti2022Light_XCiT-S12', dataset=args.source_task, threat_model='Linf')
        network = InputNormalize(network, [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    elif args.network == "clean_XCIT-S12":
        from xcit_main import xcit 
        network = xcit.xcit_small_12_p16()
        ckpt = torch.load('xcit_small_12_p16_224.pth')
        network.load_state_dict(ckpt['model'])
        network = InputNormalize(network, [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    elif args.network == "clean_resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        network = InputNormalize(network, [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    elif args.network == "clean_resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        network = InputNormalize(network, [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    network.requires_grad_(False)
    network.eval()


    # Visual Prompt
    norm = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])
    if args.vp == 'smm':
        visual_prompt = InstancewiseVisualPrompt(imgsize, args.attribute_layers, args.patch_size, args.attribute_channels, normalize=norm)
    elif args.vp == 'pad':
        visual_prompt = ExpansiveVisualPrompt(imgsize, mask=np.zeros((32, 32)), init='zero')
    elif args.vp == 'full':
        visual_prompt = FullyVisualPrompt(imgsize)



    # label_mapping method
    if args.mapping_method in ['rlm', 'flm', 'ilm']:
        mapping_sequence = torch.randperm(source_out)[:len(class_names)].cuda()
        label_mapping = LabelMappingBase(mapping_sequence, mapping_method=args.mapping_method)
    elif args.mapping_method == 'fc':
        label_mapping = linear(source_out, len(class_names))
    
    # ckpt = torch.load(os.path.join(args.save_path, 'last.pth'))
    # visual_prompt.load_state_dict(ckpt['visual_prompt_dict'])
    # label_mapping.load_state_dict(ckpt['label_mapping_dict'])
        
        
    # merge
    model = Pipeline(visual_prompt, network, label_mapping, args)
    model.s_model.requires_grad_(False)
    model.s_model.eval()
    return model

def reduce_sum(x, rank):
    total = torch.tensor(x, device=rank)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def train(net, trainloader, optimizer, lr_scheduler, scaler, attack, rank, ngpus_per_node, args, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler[0].get_last_lr()[0], 0, 0, correct, total))
    
    if args.mapping_method == 'flm' and epoch == 0:
        mapping_sequence = generate_label_mapping_by_frequency(net, trainloader, attack, rank)
        net.zero_grad()
    elif args.mapping_method == 'ilm':
        mapping_sequence = generate_label_mapping_by_frequency(net, trainloader, attack, rank)
        net.zero_grad()

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        if args.attack == 'trades':
            inputs_adv = attack(inputs, targets)
            for opt in optimizer:
                opt.zero_grad()
            with autocast():
                logits = net(torch.cat((inputs, inputs_adv), dim=0))
                logits_cln, outputs = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
                kl = torch.nn.KLDivLoss(reduction='batchmean')
                loss_rob = kl(F.log_softmax(outputs, dim=1), F.softmax(logits_cln, dim=1))
                loss_clean = torch.nn.CrossEntropyLoss()(logits_cln, targets)
                loss = loss_clean + loss_rob * args.lambda_
        else:
            inputs_adv = attack(inputs, targets)
            for opt in optimizer:
                opt.zero_grad()
            with autocast():
                outputs = net(inputs_adv)
                loss = F.cross_entropy(outputs, targets)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            with autocast():
                print(outputs)
                if torch.isnan(inputs_adv).any() or torch.isinf(inputs_adv).any():
                    assert False, "inputs"
                x = net.module.visual_prompt(inputs_adv)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    assert False, "visual_prompt"
                x = net.module.s_model(x)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    assert False, "s_model"
                x = net.module.label_mapping(x)
                print(x)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    assert False, "label_mapping"
                x = F.cross_entropy(x, targets)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    assert False, "ce"
                print(x)
            assert False, "NaN"
        
        # Accerlating backward propagation
        scaler.scale(loss).backward()
        for opt in optimizer:
            scaler.step(opt)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler[0].get_lr()[0], train_loss / (batch_idx+1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
    
    
    # scheduling
    for lr_s in lr_scheduler:
        lr_s.step()
    # Reduce loss across all processes
    avg_loss = reduce_sum(train_loss, rank).item() / ((batch_idx+1) * ngpus_per_node)
    total_sum = reduce_sum(total, rank).item()
    correct_sum = reduce_sum(correct, rank).item()
    
    return avg_loss, 100. * correct_sum / total_sum
    
def test(net, testloader, test_attack, rank, best_f, args, ngpus_per_node):
    net.eval()
    test_loss_cln, test_loss_adv = 0, 0
    correct_cln, correct_adv = 0, 0
    total = 0
    desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss_cln/(0+1), 0, correct_cln, total)) + ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss_adv / (0 + 1), 0, correct_adv, total))


    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs_adv = test_attack(inputs, targets)

        # Accerlating forward propagation
        with torch.no_grad():
            with autocast():
                # outputs = net(inputs)
                logits = net(torch.cat((inputs, inputs_adv), dim=0))
                logits_cln, logits_adv = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
                loss_cln = F.cross_entropy(logits_cln, targets)
                loss_adv = F.cross_entropy(logits_adv, targets)

            test_loss_cln += loss_cln.item()
            test_loss_adv += loss_adv.item()
            _, predicted = logits_cln.max(1)
            total += targets.size(0)
            correct_cln += predicted.eq(targets).sum().item()
            _, predicted = logits_adv.max(1)
            correct_adv += predicted.eq(targets).sum().item()

        desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss_cln / (batch_idx + 1), 100. * correct_cln / total, correct_cln, total)) + ('[Test/PGD] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss_adv / (batch_idx + 1), 100. * correct_adv / total, correct_adv, total))
        prog_bar.set_description(desc, refresh=True)
        
    avg_loss_cln = reduce_sum(test_loss_cln, rank).item() / ((batch_idx+1) * ngpus_per_node)
    total_sum = reduce_sum(total, rank).item()
    correct_sum_cln = reduce_sum(correct_cln, rank).item()  
    
    avg_loss_adv = reduce_sum(test_loss_adv, rank).item() / ((batch_idx+1) * ngpus_per_node)
    correct_sum_adv = reduce_sum(correct_adv, rank).item()  
    # Save clean acc.
    clean_acc = 100. * correct_sum_cln / total_sum

    # Save adv acc.
    adv_acc = 100. * correct_sum_adv / total_sum

    # compute acc
    f = (clean_acc + adv_acc)/2

    print('Current Accuracy is {:.2f}/{:.2f}!!'.format(clean_acc, adv_acc), rank)

    state_dict = {
        "visual_prompt_dict": net.module.visual_prompt.state_dict(),
        "last_f": f,
        "best_f": best_f,
        "last_rob": adv_acc,
        "last_acc": clean_acc,
        "label_mapping_dict": net.module.label_mapping.state_dict(),
    }
    
    if f > best_f:
        best_f = f
        state_dict['best_f'] = best_f
        if rank == 0:
            torch.save(state_dict, os.path.join(args.save_path, f'best{args.seed}.pth'))
            print('Saving best~', os.path.join(args.save_path, f'best{args.seed}.pth'))
    if rank == 0:
        torch.save(state_dict, os.path.join(args.save_path, f'last{args.seed}.pth'))
        print('Saving~', os.path.join(args.save_path, f'last{args.seed}t.pth'))
            
    return clean_acc, adv_acc, best_f

def main_worker(rank, ngpus_per_node, args):
    set_seed(args.seed + rank)
    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    gpu_list = list(map(int, args.gpu.split(',')))
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)
    
    # load data
    loaders, class_names, train_sampler = get_dataloader(args)
    
    # load model
    model = get_model(args, loaders, class_names)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(memory_format=torch.channels_last).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=[rank])
    
    if args.dataset == 'imagenet' or args.dataset == 'tiny':
        print('Fast FGSM training', rank)
        attack = attack_loader(net=model, attack='fgsm_train', eps=2/255 if args.dataset == 'imagenet' else 0.03, steps=args.steps)
    else:
        print('PGD training', rank)
        attack = attack_loader(net=model, attack=args.attack, eps=args.eps, steps=args.num_steps, alpha=args.step_size)
        
    # check fixed parameters
    for name, param in model.module.visual_prompt.named_parameters():
        assert param.requires_grad, "visual_prompt is fixed"
    for name, param in model.module.s_model.named_parameters():
        assert not param.requires_grad, "source model is not fixed"
    for name, param in model.module.label_mapping.named_parameters():
        assert param.requires_grad, "label_mapping is fixed"
             
        
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=args.wd)
    # optimizer = optim.AdamW([{'params': model.module.visual_prompt.parameters(),'lr': args.lr_vp, 'weight_decay':args.wd},
    #                          {'params': model.module.label_mapping.parameters(), 'lr': args.lr_lm, 'weight_decay':args.wd}])
    optimizer_vp = optim.AdamW(model.module.visual_prompt.parameters(), lr=args.lr_vp, weight_decay=args.wd)
    lr_scheduler_vp = torch.optim.lr_scheduler.MultiStepLR(optimizer_vp,
                                                            milestones=[30, 50],
                                                            gamma=0.1)
    optimizer = [optimizer_vp]
    lr_scheduler = [lr_scheduler_vp]

    if args.mapping_method == 'fc':
        optimizer_lm = optim.AdamW(model.module.label_mapping.parameters(), lr=args.lr_lm, weight_decay=args.wd)
    
        lr_scheduler_lm = torch.optim.lr_scheduler.MultiStepLR(optimizer_lm,
                                                            milestones=[30, 50],
                                                            gamma=0.1)
        optimizer = [optimizer_vp, optimizer_lm]
        lr_scheduler = [lr_scheduler_vp, lr_scheduler_lm]
    
    # training and testing
    os.makedirs(args.save_path, exist_ok=True)
    if rank == 0:
        logger = SummaryWriter(args.save_path)
    scaler = GradScaler()
    best_f = 0.
    test_attack = attack_loader(net=model, attack='pgd', eps=8/255, steps=20, alpha=2/255)
    for epoch in range(args.epoch):
        print('Epoch: %d' % epoch, rank)
        loaders['train'].sampler.set_epoch(epoch)
        train_loss, train_adv_acc = train(model, loaders['train'], optimizer, lr_scheduler, scaler, attack, rank, ngpus_per_node, args, epoch)
        
        if epoch % 5 == 0 or epoch+1==args.epoch:
            clean_acc, adv_acc, best_f = test(model, loaders['test'], test_attack, rank, best_f, args, ngpus_per_node)
            if rank == 0:
                logger.add_scalar("train/adv", train_adv_acc, epoch)
                logger.add_scalar("train/loss", train_loss, epoch)
                logger.add_scalar("test/clean_acc", clean_acc, epoch)
                logger.add_scalar("test/adv_acc", adv_acc, epoch)

    import pandas as pd
    model.eval()
    test_attack = attack_loader(net=model, attack='pgd', eps=4/255, steps=20, alpha=2/255)
    clean_acc, adv_acc, best_f = test(model, loaders['test'], test_attack, rank, best_f, args, ngpus_per_node)
    if rank == 0:
        try:
            df = pd.read_csv(os.path.join(args.save_path, 'res_4.csv'))
            df.loc[len(df)] = [args.seed, train_adv_acc, clean_acc, adv_acc]
        except:
            df = pd.DataFrame({'seed':[args.seed], 'train_rob':[train_adv_acc], 'test_acc':[clean_acc], 'test_rob':[adv_acc]})
        df.to_csv(os.path.join(args.save_path, 'res_4.csv'), index=False)

    test_attack = attack_loader(net=model, attack='pgd', eps=8/255, steps=20, alpha=2/255)
    clean_acc, adv_acc, best_f = test(model, loaders['test'], test_attack, rank, best_f, args, ngpus_per_node)
    if rank == 0:
        try:
            df = pd.read_csv(os.path.join(args.save_path, 'res_8.csv'))
            df.loc[len(df)] = [args.seed, train_adv_acc, clean_acc, adv_acc]
        except:
            df = pd.DataFrame({'seed':[args.seed], 'train_rob':[train_adv_acc], 'test_acc':[clean_acc], 'test_rob':[adv_acc]})
        df.to_csv(os.path.join(args.save_path, 'res_8.csv'), index=False)
    
    # final test
    # destroy process net
    dist.destroy_process_group()
        


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    # env
    p.add_argument('--gpu', default='6', type=str)
    p.add_argument('--port', default='13316', type=str)
    p.add_argument('--seed', type=int, default=0)
    
    # model parameter
    p.add_argument('--dataset', default='CIFAR10', type=str)
    p.add_argument('--source_task', default='imagenet', type=str)
    p.add_argument('--network', default="Salman2020Do_R18")
    p.add_argument('--patch_size', type=int, default=8)
    p.add_argument('--attribute_channels', type=int, default=3)
    p.add_argument('--attribute_layers', type=int, default=5)
    p.add_argument('--mapping_method', type=str, default='fc', choices=['fc', 'ilm', 'rlm', 'flm'])
    p.add_argument('--vp', type=str, default='smm', choices=['smm', 'pad', 'full'])

    # learning parameter
    p.add_argument('--lr_vp', default=1e-3, type=float)
    p.add_argument('--lr_lm', default=1e-3, type=float)
    p.add_argument('--wd', default=1e-3, type=float)
    p.add_argument('--bs', default=256, type=int)
    p.add_argument('--test_batch_size', default=256, type=float)
    p.add_argument('--epoch', default=60, type=int)

    # attack
    p.add_argument('--attack', default='pgd', type=str)
    p.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)
    p.add_argument('--eps', default=8, type=int)
    p.add_argument('--num_steps', default=10, type=int)
    p.add_argument('--lambda_', default=6, type=int)
    args = p.parse_args()
    
    # init 
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    if args.bs % ngpus_per_node != 0:
        raise "args.bs % ngpus_per_node != 0"
    args.bs = args.bs // ngpus_per_node

    # cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    
    args.eps = args.eps / 255
    args.step_size = args.eps / 4
    args.random_restarts = 1
    
    args.data_path = '/home/zsj/datasets/'
    args.results_path = '/home/zsj/reprogramming/Adv_SMM/distribute/results/'
    
    train_me = args.attack + '-' + str(args.num_steps) + '-' + str(args.lambda_)
    if args.vp == 'smm':
        args.save_path = os.path.join(args.results_path,
                                args.dataset + "_" + args.source_task + "_" + args.network + "_"  + args.mapping_method + "_" + str(args.wd) + "_"  + str(args.lr_vp)  + "_" + str(args.lr_lm)  + "_"  + str(args.attribute_channels)  + "_"  + str(
                                args.attribute_layers)  + "_"  + str(args.patch_size)+ "_" + str(int(args.eps*255)) + "_" + str(args.num_steps
                                ) + "_" + str(args.constraint) + "_" + str(args.vp) + "_" + str(train_me))
    else:
        args.save_path = os.path.join(args.results_path,
                                args.dataset + "_" + args.source_task + "_" + args.network + "_"  + args.mapping_method + "_" + str(args.wd) + "_"  + str(args.lr_vp)  + "_" + str(args.lr_lm) + "_" + str(int(args.eps*255)) + "_" + str(args.num_steps
                                ) + "_" + str(args.constraint) + "_" + str(args.vp) + "_" + str(train_me))
    
    
    torch.multiprocessing.spawn(main_worker, args=(ngpus_per_node, args,), nprocs=ngpus_per_node, join=True)