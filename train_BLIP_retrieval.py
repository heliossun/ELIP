'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
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

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from models.sched import get_lr_sched

def train(model, data_loader, optimizer, epoch,global_step, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        global_step+=1
        lr_this_step = get_lr_sched(global_step, config, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
        if config['grad_clip_norm'] != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'], norm_type=2.0)
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, global_step  


@torch.no_grad()
def compute_entropy(scores_i2t, scores_t2i,ranks_i2t,ranks_t2i):
    #print("i2t shape:",scores_i2t.shape)
    #print("t2i shape:",scores_t2i.shape)
    class_num=scores_i2t.shape[0]
    
    #generate mask to make 1 image -> 1 text, used to be 1 image for 5 caption
    # if scores_i2t.shape[0]!=scores_i2t.shape[1]:   
    oneCap4Img=[]
    for i in range(scores_i2t.shape[0]-1):
        randidx=random.randint(0,4)
        oneCap4Img.append(i*5+randidx)
    #     mask_i2t=np.zeros(scores_i2t.shape)
    #     mask_t2i=np.zeros(scores_t2i.shape)
    #     for i in oneCap4Img:
    #         mask_i2t[:,i]+=1
    #     for j in np.where(ranks_t2i < 1)[0]:  
    #         mask_t2i[j]+=1
    # else:
    #     mask_i2t=np.ones(scores_i2t.shape)
    #     mask_t2i=mask_i2t
    
    alpha_i2t=np.maximum(scores_i2t,0)+1
    alpha_i2t=alpha_i2t[:,oneCap4Img]
    s_i2t=np.sum(alpha_i2t, axis=1, keepdims=False)
    alpha_t2i=np.maximum(scores_t2i,0)+1
    s_t2i=np.sum(alpha_t2i, axis=1, keepdims=False)

    s_i2t=s_i2t[np.where(ranks_i2t < 1)[0]]
    
    s_t2i=s_t2i[np.where(ranks_t2i < 1)[0]]
    return class_num/np.array(s_i2t), class_num/np.array(s_t2i)

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0).to(device)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        topk_sim.to(device)
        topk_idx.to(device)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

def oodEval(model_without_ddp, device, OODS):
     for noise in OODS: 
        train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, noise=noise)  

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
            val_result,u_i2t,u_t2i = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)              
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
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks2 = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks2[index] = np.where(inds == txt2img[index])[0][0]

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
                    'u_t2i': float(np.mean(u_t2i)),}
    return eval_result,u_i2t,u_t2i


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
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                             va=config['va'],ta=config['ta'],evidential=config['evidential'])
    train_params=0
    freeze_params=0
    if config['adapter']:
        unfreeze=['adapter']
        for name, param in model.named_parameters():
            if any(n in name for n in unfreeze):
                #print("unfreeze",name)
                param.requires_grad=True
                train_params+=param.numel()
            else:
                #print("freeze",name)
                freeze_params+=param.numel()
                param.requires_grad=False
    log_stats = { '"trainable parameters': train_params,'freeze params': freeze_params,}
    print('trainable parameters', train_params,'freeze params', freeze_params)
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n") 
    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0
    global_step=0
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)   
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats,global_step = train(model, train_loader, optimizer, epoch, global_step, device, config) 
            if (epoch+1) % 2 == 0: 
                score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
                if utils.is_main_process():  
                    val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
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
            noises=['c','g','r']
            oodEval(model_without_ddp,device,noises)
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