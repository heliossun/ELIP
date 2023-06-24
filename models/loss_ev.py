import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=True,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

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
    #device=alpha.get_device()
    #print("current epoch:",epoch_num)
    #print('label:',1-y)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    #log_target=torch.ones(kl_alpha.shape,dtype=torch.float32, device=device)
    #kl_loss=F.kl_div(kl_alpha,log_target)
    kl_loss2 = kl_divergence(kl_alpha, num_classes)
    #print("built in kl loss:",kl_loss)
    #print('alpha:',alpha)
    #print('kl alpha:',kl_alpha)
    #print('manual kl loss: ',kl_loss2)
    kl_div = annealing_coef * kl_loss2
    return A + kl_div


def loglikelihood_loss(y, alpha):

    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):

    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = exp_evidence(output)
    alpha = evidence+1
    device = target.get_device()
    one_hot = torch.zeros((len(target), len(target)),device=device)
    one_hot[torch.arange(len(target)), target] = 1
    loss = torch.mean(mse_loss(one_hot, alpha, epoch_num, num_classes, annealing_step))
    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = exp_evidence(output)
    alpha = evidence+1
    device = target.get_device()
    one_hot = torch.zeros((len(target), len(target)),device=device)
    one_hot[torch.arange(len(target)), target] = 1
    loss = torch.mean(
        edl_loss(
            torch.log, one_hot, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss
def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = exp_evidence(output)
    alpha = evidence + 1
    device = target.get_device()
    one_hot = torch.zeros((len(target), len(target)),device=device)
    one_hot[torch.arange(len(target)), target] = 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, one_hot, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss

class EVLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            evidential=True
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.evidential = evidential
        #print("world size:",world_size)
    def forward(self, image_features, text_features, logit_scale, epoch_num):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            #print("img has NAN:",all_image_features.isnan().any())
            #print("text has NAN:",all_text_features.isnan().any())
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        #print(logits_per_image.shape)
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.evidential:
            num_classes = logits_per_image.size(0)
            total_loss = (edl_digamma_loss(logits_per_image, labels, epoch_num, num_classes , 1000)+
                        edl_digamma_loss(logits_per_text, labels, epoch_num, num_classes , 1000))/2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2
        

        #print("ce loss:", total_loss)
        #print("evidence loss:", evidence_loss)
        return total_loss
