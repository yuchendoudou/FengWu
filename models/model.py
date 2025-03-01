import torch
import torch.nn as nn
from networks.LGUnet_all import LGUnet_all
from utils.builder import get_optimizer, get_lr_scheduler
import utils.misc as utils
import time
import datetime
from pathlib import Path
import os
from collections import OrderedDict
from torch.functional import F
from utils.misc import is_dist_avail_and_initialized
from replay.replay_buff import replay_buff
import gc

LOG_SIG_MAX = 5
LOG_SIG_MIN = -8


class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0)
        self.extra_params = params.get("extra_params", {})
        self.loss_type = self.extra_params.get("loss_type", "PossLoss")
        self.whether_save_checkpoint = self.extra_params.get("whether_save_checkpoint", True)
        self.save_best = self.extra_params.get("save_best", True)
        self.save_last = self.extra_params.get("save_last", True)

        self.replay_buff_params = self.extra_params.get("replay_buff", None)
        # if self.two_step_training:
        self.checkpoint_path = self.extra_params.get('checkpoint_path', None)
        self.checkpoint_strict = self.extra_params.get("checkpoint_strict", True)


        checkpoint_dir = self.extra_params.get("checkpoint_dir", "weatherbench:s3://weatherbench/checkpoint")
        self.save_checkpoint_dir = self.extra_params.get('save_checkpoint_dir', checkpoint_dir)
    
        self.begin_epoch = 0
        self.metric_best = 1000

        if is_dist_avail_and_initialized():
            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device
        sub_model = params.get('sub_model', {})
        for key in sub_model:
            origin_key = key
            if key[-5:] == "_copy":
                key = key[:-5]
            if key == "lgunet_all":
                model = LGUnet_all(**sub_model[origin_key])
            else:
                raise NotImplementedError('Invalid model type.')
            self.model[origin_key] = model
            key = origin_key
            if self.loss_type == "Possloss":
                output_dim = self.params['sub_model'][list(self.model.keys())[0]]["out_chans"]
                img_size = self.params['sub_model'][list(self.model.keys())[0]].get("img_size", [32, 64])

                self.max_logvar = self.model[key].max_logvar = torch.nn.Parameter((torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() / 2))
                self.min_logvar = self.model[key].min_logvar = torch.nn.Parameter((-torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() * 10))


            self.model[key].to(device)
            if is_dist_avail_and_initialized():
                parallel_model = torch.nn.parallel.DistributedDataParallel(self.model[key], device_ids=[utils.get_localrank()])
            
                self.model[key] = parallel_model

            self.sub_model_name.append(key)

        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})

        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
           
            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics

        self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()


        if self.checkpoint_path is None:
            self.logger.info("finetune checkpoint path not exist")
        else:
            if isinstance(self.checkpoint_path, str):
                self.load_checkpoint(self.checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
            elif isinstance(self.checkpoint_path, dict):
                for key in self.checkpoint_path:
                    if isinstance(self.checkpoint_path[key], str):
                        self.load_checkpoint(self.checkpoint_path[key], load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False, load_parameters=key)
                    else:
                        self.load_checkpoint(key, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False, load_parameters=self.checkpoint_path[key])


        if self.loss_type == "Possloss":
            self.loss = self.Possloss


    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device)
        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def data_preprocess(self, data):
        return None, None


    def Possloss(self, pred, target, **kwargs):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = kwargs.get("inc_var_loss", True)

        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)

        # mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        # target = target.reshape(num_examples, -1)

        _, C, H, W = target.shape
        max_logvar = self.max_logvar
        min_logvar = self.min_logvar



        log_var = max_logvar - F.softplus(max_logvar - log_var)
        log_var = min_logvar + F.softplus(log_var - min_logvar)

        log_var = log_var.reshape(*(target.shape))

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var)
            var_loss = torch.mean(log_var)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2))
            total_loss = mse_loss
        total_loss += 0.01 * torch.mean(max_logvar) - 0.01 * torch.mean(min_logvar)
    

        return torch.mean(total_loss)


    def train_one_step(self, batch_data, step):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)
        else:
            raise NotImplementedError('Invalid model type.')

        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
        else:
            raise NotImplementedError('Invalid model type.')
        
        return loss


    def test_one_step(self, batch_data):
        input, target = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](input)

        data_dict = {}
        data_dict['gt'] = target
        data_dict['pred'] = predict
        return data_dict


    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)


        # test_logger = {}


        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')

        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'

        if train_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = train_data_loader

        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)
        
            # record data read time
            data_time.update(time.time() - end_time)
   
            loss = self.train_one_step(batch, step)


            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.memory_reserved() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))


    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True, 
                        load_epoch=True, load_metric_best=True, resume=False, load_parameters=[], checkpoint_strict=None):

        
               
    
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        if load_model:
            for key in self.model:
                if not isinstance(load_parameters, list):
                    load_parameters = [load_parameters,]
                if len(load_parameters) > 0:
                    if key != load_parameters[0]:
                        continue
                new_state_dict = OrderedDict()
                # model_state_dict = self.model[key]
                if resume:
                    checkpoint_key = key
                    if not (key in checkpoint_model):
                        continue
                else:
                    checkpoint_key = list(checkpoint_model.keys())[0]
                for k, v in checkpoint_model[checkpoint_key].items():
                    if is_dist_avail_and_initialized() and "module" == k[:6]:
                        name = k
                    elif is_dist_avail_and_initialized():
                        name = f"module.{k}"
                    elif "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k

                    if len(load_parameters) > 1  and not resume:
                        for load_parameter in load_parameters[1:]:
                            if load_parameter in name:
                                new_state_dict[name] = v
                                break
                    elif key in self.freeze_parameters and len(self.freeze_parameters[key]) > 0 and not resume:
                        for freeze_parameter in self.freeze_parameters[key]:
                            if freeze_parameter in name:
                                new_state_dict[name] = v
                                break
                    else:
                        new_state_dict[name] = v

                with torch.no_grad():
                    self.model[key].load_state_dict(new_state_dict, strict=self.checkpoint_strict if checkpoint_strict is None else checkpoint_strict)
                self.model[key].to(self.device)
                del new_state_dict

        if load_optimizer:
            for key in checkpoint_optimizer:
                self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        if load_scheduler:
            for key in checkpoint_lr_scheduler:
                self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']




        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best'] if 'metric_best' in checkpoint_dict else None))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)

        if utils.get_rank() == 0:
            if save_type == "save_best":
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
            else:
                checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')


        if utils.get_world_size() > 1 and utils.get_rank() == 0:
    
            torch.save(
                {
                'epoch':            epoch+1,
                'model':            {key: self.model[key].module.state_dict() for key in self.model},
                'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                'metric_best':      self.metric_best,
                # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                }, checkpoint_path
            )
        elif utils.get_world_size() == 1:
      
            torch.save(
                {
                'epoch':            epoch+1,
                'model':            {key: self.model[key].state_dict() for key in self.model},
                'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                'metric_best':      self.metric_best,
                # "max_logvar":       self.max_logvar if hasattr(self, 'max_logvar') else None,
                # "min_logvar":       self.min_logvar if hasattr(self, 'min_logvar') else None,
                }, checkpoint_path
            )


    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, max_epoches, checkpoint_savedir=None, save_ceph=False, resume=False):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        if self.replay_buff_params is not None:
            self.replay_buff = replay_buff(train_data_loader, **(self.replay_buff_params))
        if train_data_loader is not None:
            data_std = train_data_loader.dataset.get_meanstd()[1]
            if type(data_std) == torch.Tensor:
                data_std = train_data_loader.dataset.get_meanstd()[1].float()
            else:
                data_std = torch.Tensor(train_data_loader.dataset.get_meanstd()[1]).float()
        else:
            data_std = None

        if data_std.shape[-1] == 1:
            data_std = data_std.squeeze(-1).squeeze(-1)
        self.datastd = data_std.to(self.device)

        
        if utils.get_world_size() > 1:
            for key in self.model:
                utils.check_ddp_consistency(self.model[key])
        self.now_step = self.begin_epoch * len(train_data_loader)
        for epoch in range(self.begin_epoch, max_epoches):
            if train_data_loader is not None:
                train_data_loader.sampler.set_epoch(epoch)


            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            # # update lr_scheduler
            # begin_time = time.time()
            if utils.get_world_size() > 1:
                for key in self.model:
                    utils.check_ddp_consistency(self.model[key])
            metric_logger = self.test(test_data_loader, epoch)


            
            # save model
            if self.whether_save_checkpoint:
                if self.save_best and self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if self.save_last and (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')


            gc.collect()
            if is_dist_avail_and_initialized():
                torch.distributed.barrier()
        

    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        for key in self.model:
            self.model[key].eval()


        max_step = len(test_data_loader)

        if test_data_loader is None:
            data_loader = range(max_step)
        else:
            data_loader = test_data_loader


        # max_step = len(iter(test_data_loader))
        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)
        
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger

