import os
from utils.distributedsample import Distributed_replay_Sampler
from utils.misc import  dictToObj, is_dist_avail_and_initialized
from timm.scheduler import create_scheduler
import numpy as np
import utils.misc as utils



class ConfigBuilder(object):
    def __init__(self, **params):
        super(ConfigBuilder, self).__init__()
        self.model_params = params.get('model', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})

        self.logger = params.get('logger', None)
    
    def get_model(self, model_params = None):
        from models.MTS2d_model import MTS2d_model
        from models.MTS2d_finetune import MTS2d_finetune
        if model_params is None:
            model_params = self.model_params
        type = model_params.get('type', 'VQGAN')
        params = model_params.get('params', {})
        if type == 'mts2d_model':
            model = MTS2d_model(self.logger, **params)
        elif type == 'mts2d_finetune':
            model = MTS2d_finetune(self.logger, **params)
        else:
            raise NotImplementedError('Invalid model type.')
        return model
    

    
    def get_dataset(self, dataset_params = None, split = 'train'):
        from datasets.era5_npy_f32 import era5_npy_f32
        from datasets.era5_finetune_f32 import era5_finetune_f32
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, None)
        if dataset_params is None:
            return None
        if type(dataset_params) == dict:
            dataset_type = str.lower(dataset_params.get('type', 'fourcastceph'))
            if dataset_type == 'era5_npy_f32':
                dataset = era5_npy_f32(split = split, **dataset_params)
            elif dataset_type == 'era5_finetune_f32':
                dataset = era5_finetune_f32(split = split, **dataset_params)
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset
    
    def get_sampler(self, dataset, split = 'train', drop_last=False):
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        if is_dist_avail_and_initialized():
            rank = utils.get_rank()
            num_gpus = utils.get_world_size()
        else:
            rank = 0
            num_gpus = 1
        sampler = Distributed_replay_Sampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, seed=0, drop_last=drop_last)

        return sampler
   

    def get_dataloader(self, dataset_params = None, split = 'train', batch_size = None, dataloader_params = None):
        from torch.utils.data import DataLoader
        drop_last = True
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 32)
            elif split == "test":
                batch_size = self.trainer_params.get('test_batch_size', 1)
            else:
                batch_size = self.trainer_params.get('valid_batch_size', 1)
        # if split != "train":
        #     drop_last = True
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(dataset_params, split)
        if dataset is None:
            return None
        sampler = self.get_sampler(dataset, split, drop_last=drop_last)
        return DataLoader(
            dataset,
            batch_size = batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_params
        )

    def get_max_epoch(self, trainer_params = None):
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 40)


def get_optimizer(model, optimizer_params = None, resume = False, resume_lr = None):
    """
    Get the optimizer from configuration.
    
    Parameters
    ----------
    
    model: a torch.nn.Module object, the model.
    
    optimizer_params: dict, optional, default: None. If optimizer_params is provided, then use the parameters specified in the optimizer_params to build the optimizer. Otherwise, the optimizer parameters in the self.params will be used to build the optimizer;
    
    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_lr: float, optional, default: None, the resume learning rate.
    
    Returns
    -------
    
    An optimizer for the given model.
    """
    from torch.optim import AdamW
    type = optimizer_params.get('type', 'AdamW')
    params = optimizer_params.get('params', {})

    lr_list = params.get("lr_list", None)
    if lr_list is not None:
        optimizer_dict = {}
        for lr_key in lr_list:
            optimizer_dict[lr_key] = []
        optimizer_dict["default"] = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                default_param = False
                for lr_key in lr_list:
                    for param_key in lr_list[lr_key]:
                        if param_key in name:
                            optimizer_dict[lr_key].append(param)
                            default_param = True
                            break
                if default_param == False:
                    optimizer_dict["default"].append(param)
        
        parameters_to_optimize = []
        parameters_to_optimize.append({'params': optimizer_dict['default']})
        for key in optimizer_dict.keys():
            if key != "default":
                parameters_to_optimize.append({'params': optimizer_dict[key], 'lr': key})
        new_params = {}
        for key in params.keys():
            if key != "lr_list":
                new_params[key] = params[key]
        params = new_params
    else:
        if isinstance(model, list):
            parameters_to_optimize = []
            for m in model:
                parameters_to_optimize += list(filter(lambda p: p.requires_grad, m.parameters()))
        else:
            parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

    if resume:
        network_params = [{'params': parameters_to_optimize, 'initial_lr': resume_lr}]
        params.update(lr = resume_lr)
    else:
        network_params = parameters_to_optimize
    if type == 'AdamW':
        optimizer = AdamW(network_params, **params)
    else:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizer




def get_lr_scheduler(optimizer, lr_scheduler_params = None):
    """
    Get the learning rate scheduler from configuration.
    
    Parameters
    ----------
    
    optimizer: an optimizer;
    
    lr_scheduler_params: dict, optional, default: None. If lr_scheduler_params is provided, then use the parameters specified in the lr_scheduler_params to build the learning rate scheduler. Otherwise, the learning rate scheduler parameters in the self.params will be used to build the learning rate scheduler;

    resume: bool, optional, default: False, whether to resume training from an existing checkpoint;

    resume_epoch: int, optional, default: None, the epoch of the checkpoint.
    
    Returns
    -------

    A learning rate scheduler for the given optimizer.
    """

    scheduler_args = dictToObj(lr_scheduler_params)
    from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, LinearLR
    if scheduler_args.sched in ["cosine", "tanh", "step", "multistep", "plateau", "poly"]:
        scheduler, _ = create_scheduler(scheduler_args, optimizer)
    elif scheduler_args.sched == "exponential":
        begin_lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_lr = scheduler_args.min_lr
        gamma = np.exp(np.log(end_lr/begin_lr)/scheduler_args.epochs)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif scheduler_args.sched == "linear":
        begin_lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_lr = scheduler_args.min_lr
        scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=(begin_lr-end_lr)/scheduler_args.epochs,total_iters=scheduler_args.epochs)


    return scheduler
