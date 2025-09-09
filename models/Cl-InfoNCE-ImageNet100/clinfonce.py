import torch
from util.torch_dist_sum import *
from data.imagenet import *
from data.augmentation import *
import torch.nn as nn
from util.meter import *
from network.clinfo import ClInfoNCE
import time
from util.accuracy import accuracy
from math import sqrt
import math
from util.LARS import LARS
import argparse
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
import faiss 
import numpy as np 

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch, args):
    T = epoch * iteration_per_epoch + i
    warmup_iters = args.warmup_epoch * iteration_per_epoch
    total_iters = (args.epochs - args.warmup_epoch) * iteration_per_epoch

    if epoch < args.warmup_epoch:
        lr = base_lr * 1.0 * T / warmup_iters
        # print(f"learning rate {lr}")
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr, cluster_labels_all, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    graph_losses = AverageMeter('graph', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, graph_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((img1, img2), index) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch, args)
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)
        
        # print("Check")
        if cluster_labels_all is not None:
            cluster_labels = cluster_labels_all[index.long()]
        else:
            cluster_labels = None

        # compute output
        output, target, graph_loss = model(img1, img2, cluster_labels=cluster_labels)
        ce_loss = criterion(output, target)
        if graph_loss is not None:
            # print("he")
            # loss = graph_loss + ce_loss
            loss = graph_loss + ce_loss
        else:
            loss = ce_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        if graph_loss is not None:
            graph_losses.update(graph_loss.item(), img1.size(0))
        else:
            losses.update(ce_loss.item(), img1.size(0))
        top1.update(acc1[0], img1.size(0))
        top5.update(acc5[0], img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and local_rank == 0:
            progress.display(i)


def main_worker(rank, world_size, args):
    epochs = args.epochs
    warm_up = args.warmup_epoch
    # rank, local_rank, world_size = dist_init()
    torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12345',
                                world_size=world_size, rank=rank)
    local_rank = rank
    batch_size = args.batch_size_pergpu
    num_workers = 15
    base_lr = 0.075 * sqrt(batch_size * world_size)

    model = ClInfoNCE()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(rank)
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    torch.backends.cudnn.benchmark = True

    weak_aug_train_dataset = ImagenetContrastive(aug=moco_aug, max_class=1000)
    weak_aug_train_sampler = torch.utils.data.distributed.DistributedSampler(weak_aug_train_dataset)
    weak_aug_train_loader = torch.utils.data.DataLoader(
        weak_aug_train_dataset, batch_size=batch_size, shuffle=(weak_aug_train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=weak_aug_train_sampler, drop_last=True)

    train_dataset = ImagenetContrastive(aug=simclr_aug, max_class=1000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    eval_dataset = ImagenetGather()
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=eval_sampler, drop_last=False)
    


    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss()
    os.makedirs(f'checkpoints/{args.save_folder}', exist_ok=True)
    checkpoint_path = 'checkpoints/{}/clinfonce-{}.pth'.format(args.save_folder, epochs)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    print(f"Local Rank {local_rank}")
    model.train()
    for epoch in range(start_epoch, epochs):
        cluster_labels_all = perform_clustering(args, epoch, eval_loader, model, local_rank)
        print(f"cluster_labels_all {cluster_labels_all}")
        if epoch < warm_up:
            weak_aug_train_sampler.set_epoch(epoch)
            train(weak_aug_train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr, cluster_labels_all, args)
        else:
            train_sampler.set_epoch(epoch)
            train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr, cluster_labels_all, args)
        
        if ((epoch % 5) == 0) or (epoch == (epochs - 1)):
            if local_rank == 0:
                torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)

                checkpoint_path_history = 'checkpoints/{}/clinfonce-his-{}.pth'.format(args.save_folder, epoch)
                torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path_history)



# ============== clustering ==================
def compute_features(eval_loader, model, args, rank):
    print('Computing features...')
    model.eval()
    # free GPU for that batch
    torch.cuda.empty_cache()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda(rank)
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(rank, non_blocking=True)
            feat = model(images)
            features[index] = feat

    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


def perform_clustering(args, epoch, eval_loader, model, local_rank):

    if epoch>=args.warmup_epoch:

        if (epoch+1) % args.perform_cluster_epoch == 0:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args, local_rank)

            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_loader.dataset),dtype=torch.long).cuda(local_rank))
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda(local_rank))
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda(local_rank))

            if local_rank == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
                features = features.numpy()
                cluster_result = run_kmeans(features,args)  #run kmeans clustering on master node
                # save the clustering result
                if (epoch+1) % 5 == 0:
                    print("\nSaving cluster results...\n")
                    torch.save(cluster_result, 'checkpoints/{}/clusters_{}'.format(args.save_folder, epoch))

            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     


            return cluster_result['im2cluster'][0]


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d

        #if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = 1. * density/density.mean()  #scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size-pergpu', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--low_dim', type=int, default=128)
    parser.add_argument('--save_folder', type=str, default='trail1')
    parser.add_argument('--perform_cluster_epoch', type=int, default=1)
    parser.add_argument('--num_cluster', type=str, default='2500')
    args = parser.parse_args()
    print(args)
    args.num_cluster = args.num_cluster.split(";")

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()
    
