import os
import torch
from torch import nn, optim
from util import parse_bool_from_string
from nbdt.utils import progress_bar, generate_checkpoint_fname, generate_kwargs, Colors

import numpy as np
import matplotlib.pyplot as plt
from xwsol.imagenet_dict import LABEL_DIR
from nbdt.models.utils import load_state_dict, make_kwarg_optional

def training_entry_pipeline(args):
    # args is a dictionary of arguments

    print("training_entry_pipeline()")

    if args.submode is None:
        train(args)
    


def train(args):
    # args.dataset = 'ILSVRC' # 'Imagenet1000' 
    TOGGLES = [parse_bool_from_string(x) for x in args.debug_toggles] 
    """
    TOGGLES: 
    0: observe data
    1: debug with small batch size
    """

    DIRS = manage_dirs(args)

    if args.DISABLE_GPU:
        # YES. SOMETIMES WE NEED THIS FOR SILLY DEBUGGING........
        print('GPU is disabled for debugging...')
        device = 'cpu'
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('trying to use CUDA...')
        n_gpus = torch.cuda.device_count()
        print('n_gpus:',n_gpus)


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    from nbdt import data, analysis, loss, models, metrics, tree as T
    from nbdt.tree import Tree


    print("==> Preparing data..")
    dataset_train = getattr(data, args.dataset)
    dataset_test = getattr(data, args.dataset)
    # print(dataset_train) 
    """ e.g.
    <class 'nbdt.data.imagenet.Imagenet1000'>
    <class 'nbdt.data.cifar.CIFAR10'>
    """
    transform_train = dataset_train.transform_train()
    transform_test = dataset_test.transform_val()
    trainset = dataset_train(root="./dataset",split='train', transform=transform_train)  
    # the original script call it "test". It should be val, but, whatever
    testset = dataset_test(root="./dataset", split='val',transform=transform_test,)

    if TOGGLES[0]: # observing data just to be save, and then exit 
        observe_data(trainset, testset); exit()

    EARLY_STOP = None
    if TOGGLES[1]:
        test_batch_size = 4
        args.batch_size = 2
        EARLY_STOP = 8
    else: 
        test_batch_size = 200

    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=1)



    print("==> Building model..")
    if args.arch == 'ResNet50CAM':
        """
        ResNet50CAM is just ResNet50 with CAM implemented. 
        The weights are also more updated than the ResNet50 implemented in the original NBDT paper.
        """
        from xwsol.model import ResNet50CAM
        net = ResNet50CAM().backbone # we train the main resnet part.
    else:
        # the original code
        model = getattr(models, args.arch)
        # print(model) # e.g. <function ResNet18 at 0x7f56cda39ca0>
        net = model(pretrained=False, num_classes=len(trainset.classes))
    net = net.to(device)

    if device == "cuda":
        print('attempting to use DataParallel...')
        net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))


    checkpoint_fname = generate_checkpoint_fname(**vars(args))
    checkpoint_path = "./checkpoint/{}.pth".format(checkpoint_fname)
    print(f"==> Checkpoints will be saved to: {checkpoint_path}")
    
    resume_path = args.path_resume or checkpoint_path
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        if not os.path.exists(resume_path):
            print("==> No checkpoint found. Skipping...")
        elif args.arch == 'ResNet50CAM':
            print("==> resume ResNet50CAM...") # this is our implementation for WSOL research. The rest are the same as original NBDT code
            try:
                checkpoint = torch.load(checkpoint_path)
            except:
                # if checkpoint is from older pytorch version
                checkpoint = torch.jit.load(checkpoint_path)
        else:
            # from original
            checkpoint = torch.load(resume_path, map_location=torch.device(device))

        if "net" in checkpoint:
            load_state_dict(net, checkpoint["net"])
            best_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]
            Colors.cyan(
                f"==> Checkpoint found for epoch {start_epoch} with accuracy "
                f"{best_acc} at {resume_path}"
            )
        else:
            load_state_dict(net, checkpoint)
            Colors.cyan(f"==> Checkpoint found at {resume_path}")


    tree = Tree.create_from_args(args,
            classes=trainset.classes) 

    # loss
    criterion = None
    for _loss in args.loss:
        if criterion is None and not hasattr(nn, _loss):
            criterion = nn.CrossEntropyLoss()
        class_criterion = getattr(loss, _loss)
        loss_kwargs = generate_kwargs(
            args,
            class_criterion,
            name=f"Loss {args.loss}",
            globals=locals(),
        )

        """ Example for your reference:
        print(loss_kwargs)
        {
            'classes': ['n01440764', 'n01443537', ..., 'n13133613', 'n15075141'], 
            'criterion': CrossEntropyLoss(), 
            'dataset': 'ILSVRC', 
            'hierarchy': 'induced-ResNet50CAM', 
            'tree': <nbdt.tree.Tree object at 0x7f346624cbb0>, 
            'tree_supervision_weight': 1
        }
        print(class_criterion) # <class 'nbdt.loss.SoftTreeSupLoss'>
        """
        criterion = class_criterion(**loss_kwargs)
    print('==> setting up optimizer...')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # we don't use scheduler since 
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[int(3 / 7.0 * args.epochs), int(5 / 7.0 * args.epochs)]
    # )
    print('learning rate:', args.lr)

    class_analysis = getattr(analysis, args.analysis or "Noop")
    analyzer_kwargs = generate_kwargs(
        args,class_analysis,name=f"Analyzer {args.analysis}",globals=locals(),)
    analyzer = class_analysis(**analyzer_kwargs)

    metric = getattr(metrics, args.metric)()


    # Training
    @analyzer.train_function
    def train(epoch):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch, args.epochs)

        # print("\nEpoch: %d / LR: %.04f" % (epoch, scheduler.get_last_lr()[0]))
        net.train()
        train_loss = 0
        metric.clear()
        n_train = len(trainloader)

        with torch.autograd.set_detect_anomaly(True):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                metric.forward(outputs, targets)
                transform = trainset.transform_val_inverse().to(device)
                stat = analyzer.update_batch(outputs, targets, transform(inputs))

                if (batch_idx+1)% args.print_every==0 or (batch_idx+1)==n_train:
                    update_text = 'epoch:%s iter: %s/%s'%(str(epoch),str(batch_idx+1),str(n_train))
                    print('%-64s'%(str(update_text)),end='\r')

                if EARLY_STOP is not None:
                    if batch_idx>= EARLY_STOP:
                        print('\nearly stopping for training...') 
                        break

        # scheduler.step()


    @analyzer.test_function
    def test(epoch, checkpoint=True):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        metric.clear()
        with torch.no_grad():
            n_test = len(testloader)
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                if not args.disable_test_eval:
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    metric.forward(outputs, targets)
                transform = testset.transform_val_inverse().to(device)
                stat = analyzer.update_batch(outputs, targets, transform(inputs))

                if (batch_idx+1)%4==0 or (batch_idx+1)==n_test:
                    update_text = 'epoch:%s iter: %s/%s'%(str(epoch),str(batch_idx+1),str(n_test))
                    print('%-64s'%(str(update_text)),end='\r')

                if EARLY_STOP is not None: # FOR DEBUGGING
                    if batch_idx>= EARLY_STOP: 
                        print('\nearly stopping for testing...') 
                        break

        # Save checkpoint.
        acc = 100.0 * metric.report()
        if  True: # acc > best_acc and checkpoint:
            Colors.green(f"Saving to {checkpoint_fname} ({acc})..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(state, f"./checkpoint/{checkpoint_fname}.pth")

            if acc >= best_acc and checkpoint:
                torch.save(state, "./checkpoint/%s.best.%s.pth"%(str(checkpoint_fname),str(epoch)))
                best_acc = acc

        print(
            "Accuracy: {}, {}/{} | Best Accurracy: {}".format(
                acc, metric.correct, metric.total, best_acc
            )
        )


    import time
    start = time.time()
    if args.eval:
        with analyzer.epoch_context(0):
            test(0, checkpoint=False)
    else:
        for epoch in range(start_epoch, args.epochs):
            with analyzer.epoch_context(epoch):
                train(epoch)
                test(epoch)

    end = time.time()
    elapsed = end - start
    print('\n\nTraining time taken %s [s] %s [min] %s [hr] '%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)),str(round(elapsed/3600.,1)) ))    

def manage_dirs(args):
    """
    Add more stuffs here if necessary
    """
    DIRS = {}
    DIRS['ROOT_DIR'] = args.ROOT_DIR
    return DIRS


def observe_data(trainset, testset):
    def reformatimg(x):
        x = x.clone().detach().cpu().numpy().transpose(1,2,0)
        x = (x-np.min(x))
        x = x/np.max(x)
        return x
    def reformatlabel(y):
        y = str(LABEL_DIR[str(y)]["label"])
        if len(y)>18:
            y = y[:18] + '...'
        return y

    # print(trainset.class_to_idx)
    a, a_label = trainset.__getitem__(0)
    a = a.clone().detach().cpu().numpy()
    b, b_label= testset.__getitem__(0)
    b = b.clone().detach().cpu().numpy()
 
    print('a',a.shape, a_label) # e.g. a (3, 224, 224) 0
    print('b',b.shape, b_label) # e.g. a (3, 224, 224) 0
    print('np.max(a):',np.max(a))

    train_len = trainset.__len__() 
    test_len = testset.__len__()
    print(train_len, test_len)

    train_indices = np.random.randint(0,train_len,size=(9,))    
    test_indices = np.random.randint(0,test_len,size=(9,))    

    plt.figure(figsize=(14,6))
    for i in range(1,1+9):
        a, a_label = trainset.__getitem__(train_indices[i-1])
        b, b_label= testset.__getitem__(test_indices[i-1])
        
        a = reformatimg(a)
        b = reformatimg(b)
        plt.gcf().add_subplot(3,6,i)
        plt.gca().imshow(a)
        plt.gca().set_xlabel('a:%s'%(reformatlabel(a_label)))

        plt.gcf().add_subplot(3,6,i+9)
        plt.gca().imshow(b)
        plt.gca().set_xlabel('b:%s'%(reformatlabel(b_label)))


        if i>1:
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()
        