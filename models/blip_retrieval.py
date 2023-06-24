from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -100, 100))
    #return torch.exp(y)

def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes):
    device = alpha.get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A+kl_div

def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    device = target.get_device()
    #one_hot = torch.zeros((len(target), len(target)),device=device)
    #one_hot[torch.arange(len(target)), target] = 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence+1
    device = target.get_device()
    #one_hot = torch.zeros((len(target), len(target)),device=device)
    #one_hot[torch.arange(len(target)), target] = 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = './configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 256,
                 momentum = 0.995,
                 negative_all_rank = False,
                 adapter=False
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer,adapter=adapter)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False,adapter=adapter)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        # self.visual_encoder_m, vision_width = create_vit(vit,image_size,adapter=False)              
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False,adapter=False)    
        # self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        # self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                     [self.vision_proj,self.vision_proj_m],
        #                     [self.text_encoder,self.text_encoder_m],
        #                     [self.text_proj,self.text_proj_m],
        #                    ]       
        # self.copy_params()

        # # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queu", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, alpha, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)        
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()    
        #sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        #print(sim_targets)
        sim_targets=pos_idx #evloss
        #print("sim target:",(sim_targets==0.5).sum(dim=1))
        # get momentum features
        with torch.no_grad():
            #self._momentum_update()
            #image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = image_feat.clone().detach()
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            #text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                #return_dict = True, mode = 'text')    
            text_feat_m = text_feat.clone().detach()
            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            #sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            #sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            #sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            #sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        
        
        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
        #print(sim_i2t.shape)      
        #print(sim_t2i.shape)              
        #loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        #loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
        num_classes=sim_i2t.shape[1]
        #print("sim matrix shape: ",sim_i2t.shape, sim_t2i.shape)
        loss_i2t=edl_digamma_loss(sim_i2t, sim_targets, 1, num_classes , 1000)
        loss_t2i=edl_digamma_loss(sim_t2i, sim_targets, 1, num_classes , 1000)
        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)        

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
        
        
        if self.negative_all_rank:    
            # compute sample similarity
            with torch.no_grad():                
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)        

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            with torch.no_grad():                
                mask = torch.eq(idx, idx.t())
                
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])            
            
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        return loss_ita, loss_itm 
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        
        #print(image_feats.shape)
        batch_size = image_feats.shape[0]
        
        ptr = int(self.ptr_queu)
        assert self.queue_size % batch_size == 0  # for simplicity
        #print(ptr, batch_size)
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queu[0] = ptr  


def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        #print("missing keys:")
        #print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
