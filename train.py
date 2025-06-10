import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger


#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("train", args.run_dir, utils.get_rank(), filename='iter.log', resume=args.resume)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    logger.info('Building dataloaders ...')

    train_dataloader = builder.get_dataloader(split = 'train')
    logger.info('Train dataloaders build complete')
    test_dataloader = builder.get_dataloader(split = 'test')
    logger.info('Test dataloaders build complete')

    model_params = args.cfg_params['model']['params']
    if 'train_steps' in model_params:
        steps_per_epoch = model_params['train_steps']
    else:
        steps_per_epoch = len(train_dataloader)


    # steps_per_epoch = len(train_dataloader)

    if 'lr_scheduler' in model_params:
        lr_scheduler_params = model_params['lr_scheduler']
        for key in lr_scheduler_params:
            if 'by_step' in lr_scheduler_params[key]:
                if lr_scheduler_params[key]['by_step']:
                    for key1 in lr_scheduler_params[key]:
                        if "epochs" in key1:
                            lr_scheduler_params[key][key1] *= steps_per_epoch
    
    # build model
    logger.info('Building models ...')
    model = builder.get_model()

    model_checkpoint = os.path.join(args.run_dir, 'checkpoint_latest.pth')
    if args.resume:
        model.load_checkpoint(model_checkpoint, resume=True)

    for key in model.model:
        params = [p for p in model.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])


        logger.info("params {key}: {cnt_params}".format(key=key, cnt_params=cnt_params))

    logger.info('begin training ...')

    model.trainer(train_dataloader, test_dataloader, builder.get_max_epoch(), checkpoint_savedir=args.run_dir, resume=args.resume)

    
def main(args):
    print(args.desc)
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
    desc = f'world_size{args.world_size:d}'

    if args.desc is not None:
        desc += f'-{args.desc}'

    alg_dir = args.cfg.split("/")[-1].split(".")[0]
    alg_dir_head = args.cfg.split("/")[-2]
    # args.outdir = args.outdir + "/" + alg_dir_head + "/" + alg_dir
    args.outdir = args.outdir + "/" + alg_dir
    run_dir = os.path.join(args.outdir, f'{desc}')
    relative_checkpoint_dir = alg_dir + "/" + f'{desc}'
    args.relative_checkpoint_dir = relative_checkpoint_dir
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    train_config_file = os.path.join(run_dir, 'training_options.yaml')

    if (not args.resume) or args.resume_from_config or (not os.path.exists(train_config_file)):
        print("load yaml from config")
        with open(args.cfg, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        
    else:
        print("load yaml from resume")
        with open(train_config_file, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        del_keys = []
        for key in cfg_params:
            if key in args:
                del_keys.append(key)
        for key in del_keys:
            del cfg_params[key]
    
    # cfg_params['dataloader']['num_workers'] = args.per_cpus
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len

    if args.rank == 0:
        with open(os.path.join(run_dir, 'training_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    print('Launching processes...')
    subprocess_fn(args)
    print('Done!')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size',     type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--pipeline_model_parallel_size',   type = int,     default = 1,                                            help = 'pipeline_model_parallel_size')
    parser.add_argument('--resume',                         action = "store_true",                                                  help = 'resume')
    parser.add_argument('--resume_from_config',             action = "store_true",                                                  help = 'resume from config')
    parser.add_argument('--seed',                           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',                           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',                     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',                       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--init_method',                    type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',                         type = str,     default='./IF_WKP',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',                      type = str,     default = os.path.join('configs', 'default.yaml'),      help = 'path to the configuration file')
    parser.add_argument('--desc',                           type=str,       default='STR',                                          help = 'String to include in result dir name')


    args = parser.parse_args()

    main(args)