import random

if __name__ == '__main__':
    import argparse
    import json
    import os
    import sys

    import torch.cuda
    import atexit

    from defaults import *
    from utils import *
    from models import *
    from datasets import *
    from training import *

    # To allow entirely deterministic behaviour
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_SILENT"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=int, default=0)
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--epoch_weighting', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--user', type=str, default='cuda:0')
    parser.add_argument('--preset', type=str, default='')
    parser.add_argument('--epochs', type=int, nargs='+', default=20)
    parser.add_argument('--data', type=str, default='HappyFaces')
    parser.add_argument('--classes', nargs='+', type=str, default=('happy', 'sad'))
    parser.add_argument('--nfeats', type=int, default=2)
    parser.add_argument('--multiply', type=float, default=1)
    parser.add_argument('--labeled_scale', type=float, default=1)
    parser.add_argument('--distinct_scale', type=float, default=1)
    parser.add_argument('--class_certainty_scale', type=float, default=1)
    parser.add_argument('--lr', type=float, nargs='+', default=0.0004)
    parser.add_argument('--lpft', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--labeled_batch_size', type=int, default=50)
    parser.add_argument('--unlabeled_batch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--image_log_frequency', type=int, default=10)
    parser.add_argument('--mix_rates', nargs=2, type=float, default=(-1, 0.5))
    parser.add_argument('--optimizer', type=str, default='torch.optim.Adam')
    parser.add_argument('--labeled_criterion', type=str, default='nn.BCELoss()')
    parser.add_argument('--unlabeled_criterion', type=str, default='nn.BCELoss(reduction="none")')
    parser.add_argument('--models', type=str, default='PretrainedResnetClassifier(18,2)')
    parser.add_argument('--trainer', type=str, default='DistinctTrainer')
    parser.add_argument('--extra_vars', type=str, default="0 1 2 3")
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--load', type=str, default="")
    parser.add_argument('--saliency', type=str, default="")
    parser.add_argument('--saliency_step', type=int, default=-1)
    parser.add_argument('--limit', type=int, default=-1)

    args = parser.parse_args()

    args = vars(args)

    if args['preset'] != '':
        with open(experiment_presets_file) as p:
            presets = json.load(p)
        if args['preset'] in presets.keys():
            print('Loading experiment preset {}...'.format(args['preset']))
            args.update(presets[args['preset']])
        else:
            print('No preset named {} found in {}. Sorry, bye!'.format(args['preset'], experiment_presets_file))
            exit()

    for arg in ['lr', 'epochs']:
        if isinstance(args[arg], list) and len(args[arg]) == 1:
            args[arg] = args[arg][0]

    if not torch.cuda.is_available() or args['user'] == 'cpu':
        print('Using cpu...')
        args.update({'device': torch.device('cpu')})
    else:
        args.update({'device': torch.device(args['user'])})

    if args['seed'] == -1:
        args['seed'] = random.randint(0, 10000)

    if args['batch_size'] != 0:
        args['labeled_batch_size']   = args['batch_size']
        args['unlabeled_batch_size'] = args['batch_size']

    set_seed(args['seed'])

    def exit_handler():
        print('Finishing run...')
#        run.finish()
        print('Done!')


    atexit.register(exit_handler)

    print('Building experiment from arguments...')
    args['models'] = [eval(args['models'])]
    
    if args['load'] != '':
        load_params(args['models'], args['load'])

    args['extra_vars'] = '[' + args['extra_vars'].replace(' ', ',') + ']'

    args['container'] = Container(dr=args['data'],
        labeled_mix_rate=args['mix_rates'][0], unlabeled_mix_rate=args['mix_rates'][1],
        labeled_batch_size=args['labeled_batch_size'], unlabeled_batch_size=args['unlabeled_batch_size'])
    args['labeled_criterion'] = eval(args['labeled_criterion'])
    args['unlabeled_criterion'] = eval(args['unlabeled_criterion'])
    args['optim'] = eval(args['optimizer'])
    args['name'] = "default_name"



    trainer = eval(args['trainer'])(**args)
    if args['skip_train']:
        trainer.evaluate()
    else:
        trainer.train(args['epochs'])
    if args["test"]:
        trainer.evaluate(test=True)

