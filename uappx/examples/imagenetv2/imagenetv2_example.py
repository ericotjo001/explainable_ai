

def run_imagenetv2_example(args, dargs, parser):
    print('imagenetv2_example...')

    parser.add_argument('--submode', default='hyper', type=str, help=None)
    parser.add_argument('--kwidth', default=16, type=int, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if dargs['submode'] == 'hyper':
        from .imagenetv2_hyper import imagenetv2_hyper_
        imagenetv2_hyper_()
    else:
        raise NotImplementedError('invalid submode?')