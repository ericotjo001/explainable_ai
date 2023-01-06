import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--data', default=None, type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if args.data == 'imagenet':
        from src import entry_imagenet
        entry_imagenet(parser)
    elif args.data == 'chestxray_pneu':
        from src import entry_chestxray
        entry_chestxray(parser)
    elif args.data == 'chestxray_covid':
        from src import entry_chestxray_covid
        entry_chestxray_covid(parser)
    elif args.data == 'creditcardfraud':
        from src import entry_creditcardfraud
        entry_creditcardfraud(parser)
    elif args.data == 'drybean':
        from src import entry_drybean
        entry_drybean(parser)
    else:
        raise NotImplementedError()