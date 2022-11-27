import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if args.mode=='gputest':
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

    elif args.mode == 'prepare_data':
        parser.add_argument('--PROJECT_NAME', default='project00', type=str, help=None)
        parser.add_argument('--CKPT_FOLDER_DIR', default='checkpoint', type=str, help=None)
        parser.add_argument('--n_classes', default=10, type=int, help=None)
        parser.add_argument('--n_shards', default=4, type=int, help=None)
        parser.add_argument('--n_per_shard', default=16, type=int, help=None)
        parser.add_argument('--data_mode', default='train', type=str, help=None)
        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary

        from src.data import prepare_data
        prepare_data(dargs)

    elif args.mode == 'train_val':
        parser.add_argument('--PROJECT_NAME', default='project00', type=str, help=None)
        parser.add_argument('--CKPT_FOLDER_DIR', default='checkpoint', type=str, help=None)
        parser.add_argument('--n_classes', default=3, type=int, help=None)
        parser.add_argument('--batch_size', default=16, type=int, help=None)
        parser.add_argument('--n_epoch', default=1, type=int, help=None)
        parser.add_argument('--min_epoch', default=-1, type=int, help=None)

        parser.add_argument('--model_name', default='model', type=str, help=None)
        parser.add_argument('--loss', default='CE', type=str, help=None)
        parser.add_argument('--mab_regularization', default=1., type=float, help=None)
        

        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary

        from src.train import train_val_mabcam, sample_heatmaps
        train_val_mabcam(dargs)
        sample_heatmaps(dargs)

    elif args.mode =='eval':
        parser.add_argument('--PROJECT_NAME', default='project00', type=str, help=None)
        parser.add_argument('--CKPT_FOLDER_DIR', default='checkpoint', type=str, help=None)
        parser.add_argument('--n_classes', default=3, type=int, help=None)
        parser.add_argument('--model_name', default='model', type=str, help=None)

        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary

        from src.eval import eval_mabcam
        eval_mabcam(dargs)

    elif args.mode == 'visualization':
        parser.add_argument('--PROJECT_NAME', default='project01', type=str, help=None)
        parser.add_argument('--CKPT_FOLDER_DIR', default='checkpoint', type=str, help=None)
        parser.add_argument('--model_names', nargs='+', default=['mabmodel_10_1','mabmodel_10_2']) 
        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary

        from src.visualization import visualization_
        visualization_(dargs)


    elif args.mode == 'eval_and_gallery':
        parser.add_argument('--PROJECT_NAME', default='project01', type=str, help=None)
        parser.add_argument('--CKPT_FOLDER_DIR', default='checkpoint', type=str, help=None)
        parser.add_argument('--n_classes', default=3, type=int, help=None)
        parser.add_argument('--model_name', default='model', type=str, help=None)
        
        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary

        from src.visualization import eval_and_gallery_
        eval_and_gallery_(dargs)