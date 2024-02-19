import os
from nbdt.utils import DATASETS, METHODS, Colors, fwd
from nbdt.hierarchy import (
    print_graph_stats,
    assert_all_wnids_in_graph,
    test_hierarchy,
    generate_vis_fname,
    generate_hierarchy_vis_from,
)
from nbdt.graph import (
    build_induced_graph,
    get_graph_path_from_args,
)    
from nbdt.thirdparty.wn import (
    # get_wnids,
    # synset_to_wnid,
    # wnid_to_name,
    get_wnids_from_dataset,
)
from nbdt.thirdparty.nx import (
    write_graph,
    # get_roots,
    # get_root,
    read_graph,
    # get_leaves,
    # get_depth,
)
import torchvision.models as models



def inducegraph(args):
    print('inducegraph()')
    args.method = 'induced' # we only use induced method
    args.vis_leaf_images = None 

    wnids = get_wnids_from_dataset(args.dataset)

    if args.arch=='ResNet18':
        print('ResNet18')
        model = models.resnet18(pretrained=True)
    elif args.arch=='ResNet50':
        print('ResNet50')
        model = models.resnet50(pretrained=True)  
    elif args.arch=='ResNet50CAM':
        # this is the model we will use for our WSOL study 
        from xwsol.model import ResNet50CAM
        model = ResNet50CAM().backbone      
    else:
        raise NotImplementedError()

    G = build_induced_graph(
        wnids,
        dataset=args.dataset,
        checkpoint=None,
        # model=args.arch,
        # linkage=induced_linkage,
        # affinity=induced_affinity,
        # branching_factor=branching_factor,
        state_dict=model.state_dict() if model is not None else None,
    )
    print_graph_stats(G, "matched")
    assert_all_wnids_in_graph(G, wnids)

    path = get_graph_path_from_args(
        dataset=args.dataset,
        method=args.method,
        # seed=seed,
        # branching_factor=branching_factor,
        # extra=extra,
        # no_prune=no_prune,
        # fname=fname,
        # path=path,
        # single_path=single_path,
        # induced_linkage=induced_linkage,
        # induced_affinity=induced_affinity,
        # checkpoint=checkpoint,
        arch=args.arch,
    )
    print(path)
    write_graph(G, path)

    Colors.green("==> Wrote tree to {}".format(path))

    test_hierarchy(args)
    generate_hierarchy_vis(args)


# Adapted from nbdt.hierarchy 
def generate_hierarchy_vis(args):
    path_hie = get_graph_path_from_args(**vars(args))
    print("==> Reading from {}".format(path_hie))
    G = read_graph(path_hie)

    pathfolder = path_hie.split('/')
    pathfolder = os.path.join(*pathfolder[1:-1])

    path_html = f"{generate_vis_fname(**vars(args))}.html"
    path_html = './%s/%s'%(str(pathfolder),str(path_html))

    kwargs = vars(args)

    dataset = None
    if args.dataset and args.vis_leaf_images:
        cls = getattr(data, kwargs.pop('dataset'))
        dataset = cls(root="./data", train=False, download=True)

    kwargs.pop('dataset', '')
    kwargs.pop('fname', '')
    return generate_hierarchy_vis_from(
        G, dataset, path_html, verbose=True, **kwargs
    )

"""
Adapted from the following file nbdt-hierarchy, which you will find in .local/bin during installation 

from nbdt.hierarchy import generate_hierarchy, test_hierarchy, generate_hierarchy_vis
from nbdt.graph import get_parser
from nbdt.utils import maybe_install_wordnet


def main():
    maybe_install_wordnet()
    
    parser = get_parser()
    args = parser.parse_args()

    generate_hierarchy(**vars(args))
    test_hierarchy(args)
    generate_hierarchy_vis(args)


if __name__ == '__main__':
    main()

"""