
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.elip_retrieval import elip_retrieval,tokenize
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from models.loss_ev import EVLoss
from models.sched import get_lr_sched
NUM_ACCUMULATION_STEPS = 1

def build_loss(config):
    loss = EVLoss(
        local_loss=config['local_loss'],
        gather_with_grad=config['gather_with_grad'],
        cache_labels=True,
        rank=utils.get_rank(),
        world_size=utils.get_world_size(),
        use_horovod=False,
        evidential=config['evidential'])
    
    return loss

def train(model,loss, data_loader, optimizer, epoch, global_step, device, config, args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        caption = tokenize(caption).to(device,non_blocking=True)
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
        image_features, text_features, logit_scale = model(image, caption)  
        optimizer.zero_grad()
        total_loss = loss(image_features, text_features, 100, epoch+1)              
        total_loss = total_loss / NUM_ACCUMULATION_STEPS
        total_loss.backward()
        if ((i+1)% NUM_ACCUMULATION_STEPS ==0):
            global_step += 1
            lr_this_step = get_lr_sched(global_step, config, args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            if config['grad_clip_norm'] != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'], norm_type=2.0)
            optimizer.step()  
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()},global_step

@torch.no_grad()
def compute_entropy(scores_i2t, scores_t2i,ranks_i2t,ranks_t2i):
    class_num=scores_i2t.shape[0]
    
    #generate mask to make 1 image -> 1 text, used to be 1 image for 5 caption
    oneCap4Img=[]
    for i in range(scores_i2t.shape[0]-1):
        randidx=random.randint(0,4)
        oneCap4Img.append(i*5+randidx)
    
    alpha_i2t=np.exp(scores_i2t*100)+1
    alpha_i2t=alpha_i2t[:,oneCap4Img]
    s_i2t=np.sum(alpha_i2t, axis=1, keepdims=False)
    alpha_t2i=np.exp(scores_t2i*100)+1
    s_t2i=np.sum(alpha_t2i, axis=1, keepdims=False)

    s_i2t=s_i2t[np.where(ranks_i2t < 1)[0]]
    
    s_t2i=s_t2i[np.where(ranks_t2i < 1)[0]]
    return class_num/np.array(s_i2t), class_num/np.array(s_t2i)

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    model.eval() 
    print('Computing features for evaluation...')
    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text= tokenize(text).to(device) 
        text_embed = model.encode_text(text) 
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        text_embeds.append(text_embed)   
    text_embeds = torch.cat(text_embeds,dim=0)
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_embed = model.encode_image(image)         
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        image_embeds.append(image_embed)  
    image_embeds = torch.cat(image_embeds,dim=0)
    sims_i2t = image_embeds @ text_embeds.t()
    sims_t2i = text_embeds @ image_embeds.t()
    return sims_i2t.cpu().numpy(), sims_t2i.cpu().numpy()

@torch.no_grad()
def oodEval(model_without_ddp, device, text_noises, image_noises):
    def one_eval(train_dataset, val_dataset, test_dataset):
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                            num_workers=[4,4,4],
                                                            is_trains=[True, False, False], 
                                                            collate_fns=[None,None,None])
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
        
        if utils.is_main_process():  
            val_result,u_i2t,u_t2i,top1_i2t,top1_t2i = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt,val_loader)  
            
            print("current noise: ",noise, val_result)              
            log_stats = {**{f'val_{k}': v for k, v in val_result.items()},}
            with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")     
            f = open(os.path.join(args.output_dir,f"{noise}_i2t.txt"), "w")
            for i in u_i2t:
                f.write('{:.2f}\n'.format(float(i)))
            f.close()
            f = open(os.path.join(args.output_dir,f"{noise}_t2i.txt"), "w")
            for i in u_t2i:
                f.write('{:.2f}\n'.format(float(i)))
            f.close()    
            f = open(os.path.join(args.output_dir,f"top1R_{noise}_i2t.txt"), "w")
            for i in top1_i2t:
                f.write(f'{i}\n')
            f.close()    
            f = open(os.path.join(args.output_dir,f"top1R_{noise}_t2i.txt"), "w")
            for i in top1_t2i:
                f.write(f'{i}\n')
            f.close()    
    
    for noise in image_noises: 
        "<<<<<evaluating image noises>>>>>"
        train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, 
                                                                  text_noise= 'clean', img_noise= noise)  
        one_eval(train_dataset, val_dataset, test_dataset)
    # for noise in text_noises: 
    #     "<<<<<evaluating text noises>>>>>"
    #     train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, 
    #                                                               text_noise= noise, img_noise= 'clean')  
    #     one_eval(train_dataset, val_dataset, test_dataset)

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, data_loader):
    
    #Images->Text 
    texts = data_loader.dataset.text 
    imgs = data_loader.dataset.image
    ranks = np.zeros(scores_i2t.shape[0])
    top1_i2t=[]
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1_i2t.append((imgs[index],texts[inds[0]]))
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    #Text->Images 
    ranks2 = np.zeros(scores_t2i.shape[0])
    top1_t2i=[]
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks2[index] = np.where(inds == txt2img[index])[0][0]
        top1_t2i.append((texts[index],imgs[inds[0]]))
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks2 < 1)[0]) / len(ranks2)
    ir5 = 100.0 * len(np.where(ranks2 < 5)[0]) / len(ranks2)
    ir10 = 100.0 * len(np.where(ranks2 < 10)[0]) / len(ranks2)        

    u_i2t,u_t2i = compute_entropy(scores_i2t,scores_t2i,ranks,ranks2)
            
    
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'u_i2t': float(np.mean(u_i2t)),
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean,
                    'u_t2i': float(np.mean(u_t2i))}
    return eval_result,u_i2t,u_t2i,top1_i2t,top1_t2i

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  
    args.total_samples= len(train_dataset)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
    model= elip_retrieval(pretrain_clip=config['pretrain_clip'],pretrained=config['pretrained'],device=device,config=config)
    train_params=0
    freeze_params=0
    unfreeze=['adapter']
    for name, param in model.named_parameters():
        if any(n in name for n in unfreeze):
            param.requires_grad=True
            train_params+=param.numel()
        else:
            freeze_params+=param.numel()
            param.requires_grad=False
    log_stats = { '"trainable parameters': train_params,'freeze params': freeze_params,}
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")   
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        print("distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    loss = build_loss(config)
    best = 0
    best_epoch = 0
    global_step=0
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats, global_step = train(model,loss, train_loader, optimizer, epoch, global_step, device, config, args)   
            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
            print("score i2t:",score_val_i2t.shape)
            print("score t2i:",score_val_t2i.shape)
            if utils.is_main_process():  
        
                val_result,_,_,_,_ = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt, val_loader)  
                print(val_result)
                                    
                if val_result['r_mean']>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']        
                    best_epoch = epoch             
                if args.evaluate:                
                    log_stats = {**{f'val_{k}': v for k, v in val_result.items()},                
                                }
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")     
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_result.items()}, 
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")   
                       
        if args.evaluate: 
            print(">>>Evaluating ID & OOD cases<<<")
            text_noises=['formal']
            img_noises=['jpeg','snow','zoom-blur']
            #img_noises=['clean']
            oodEval(model_without_ddp,device,text_noises,img_noises)
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)