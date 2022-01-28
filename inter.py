import pandas

from align import *
from helpers import validate_graph


def multi(dataset="mnist", analyze=False):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()

    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.secondary = True
    args.version = args.version + '_secondary'
    best_timestamp = get_best_timestamp(args)
    info2 = InfoMan(args, timestamp=best_timestamp)
    info2.split = "train"
    info2.load_checkpoint()

    align = MultiAlignment([info, info2], different_inputs=False)
    g1 = align.build_graph(id="same_inputs")

    align = MultiAlignment([info2, info2], different_inputs=True)
    g2 = align.build_graph(id="different_real_inputs")
    acc, _, _, _ = validate_graph(g1, g2)
    pass

    if analyze:
        for e in g2.g.edges:
            a, b = e
            if 20 <= a[1] < 20 + 32 and b[1] >= 20 + 32:
                print(e)

        for adx in range(20, 52):
            for bdx in range(52, 84):
                for a in ((True, adx), (False, adx)):
                    for b in ((True, bdx), (False, bdx)):
                        precision = g2.get_precision(a, b)
                        recall = g2.get_recall(a, b)
                        print(f"({a}, {b}) {precision=} {recall=}")

        # filter edges
        inter_model_accs = {}
        for e, tv in acc.items():
            a, b = e
            if 20 <= a[1] < 20 + 32 and b[1] >= 20 + 32:
                inter_model_accs[e] = tv

        for e in inter_model_accs:
            a, b = e
            max_recall = 0
            max_recall_edge = None
            for c in list(g1.g.successors(a)):
                rr = g1.g[a][c]["recall"]
                if rr > max_recall:
                    max_recall_edge = (a, c)
                    max_recall = rr
            for c in list(g1.g.predecessors(a)):
                rr = g1.g[c][a]["recall"]
                if rr > max_recall:
                    max_recall_edge = (c, a)
                    max_recall = rr
            print(f"{max_recall_edge=}, {max_recall=}")

            for c in list(g2.g.successors(b)):
                rr = g2.g[b][c]["recall"]
                if rr > max_recall:
                    max_recall_edge = (b, c)
                    max_recall = rr
            for c in list(g2.g.predecessors(b)):
                rr = g2.g[c][b]["recall"]
                if rr > max_recall:
                    max_recall_edge = (c, b)
                    max_recall = rr
            print(f"{max_recall_edge=}, {max_recall=}")

    return


def mnist_and_mnist2():
    configs = get_configs()
    args = configs["mnist"]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()

    args = configs["mnist2"]
    args = get_parser().parse_args(args.split())
    args.secondary = True
    args.version = args.version + '_secondary'
    best_timestamp = get_best_timestamp(args)
    info2 = InfoMan(args, timestamp=best_timestamp)
    info2.split = "train"
    info2.load_checkpoint()

    align = MultiAlignment([info, info2], different_inputs=False)
    g1 = align.build_graph(id="mnist_mnist2")
    pass

    for e in g1.g.edges:
        a, b = e
        if 12 <= a[1] < 44 and b[1] >= 44:
            print(f"{e}: {g1.g[a][b]}")

    for e in g1.g.edges:
        a, b = e
        if a[1] < 10 and b[1] >= 44:
            print(f"{e}: {g1.g[a][b]}")


def real_and_unreal(dataset="mnist"):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()

    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.secondary = True
    args.version = args.version + '_secondary'
    best_timestamp = get_best_timestamp(args)
    info2 = InfoMan(args, timestamp=best_timestamp)
    info2.split = "train"
    info2.load_checkpoint()

    align = MultiAlignment([info, info2], different_inputs=False)
    align.set_random(False, True)
    g1 = align.build_graph(id="real_unreal")
    pass

    targets = g1.num_preds - g1.num_latent_predicates
    latent = g1.num_latent_predicates // 2
    count = 0
    for e in g1.g.edges:
        a, b = e
        if targets <= a[1] < targets + latent and b[1] >= targets + latent:
            print(f"{e}: {g1.g[a][b]}")
            count += 1
    return count


def all_real_unreal():
    df = {"dataset": [],
          "edges": []}
    for ds in get_configs():
        cnt = real_and_unreal(ds)
        df["dataset"].append(ds)
        df["edges"].append(cnt)
    df = pandas.DataFrame(df)
    path = save_dir / "figures" / "real_unreal.csv"
    with path.open('w') as f:
        df.to_csv(f)
    return df


def all_real_unreal_results():
    path = save_dir / "figures" / "real_unreal.csv"
    with path.open('r') as f:
        df = pandas.read_csv(f)
    return df


def all_inter_experiments(enable=None):
    for ds in get_configs():
        if enable:
            if ds not in enable:
                continue
        inter_experiments(ds)

rerun=False

def inter_experiments(dataset):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.load_checkpoint()

    info_copy = InfoMan(args, timestamp=best_timestamp)
    info_copy.load_checkpoint()

    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.secondary = True
    args.version = args.version + '_secondary'
    best_timestamp = get_best_timestamp(args)
    info2 = InfoMan(args, timestamp=best_timestamp)
    info2.load_checkpoint()

    if not rerun:
        align = MultiAlignment([info, info2], different_inputs=False)
        info.split, info2.split = "train", "train"
        align.build_graph(id="two_models_same_inputs")
    align = MultiAlignment([info, info2], different_inputs=False)
    info.split, info2.split = "test", "test"
    align.build_graph(id="two_models_same_inputs")

    if not rerun:
        align = MultiAlignment([info, info_copy], different_inputs=True)
        info.split, info_copy.split = "train", "train"
        align.build_graph(id="same_model_different_real_inputs")
    align = MultiAlignment([info, info_copy], different_inputs=True)
    info.split, info_copy.split = "test", "test"
    align.build_graph(id="same_model_different_real_inputs")

    if not rerun:
        align = MultiAlignment([info, info2], different_inputs=True)
        info.split, info2.split = "train", "train"
        align.build_graph(id="two_models_different_real_inputs")
    align = MultiAlignment([info, info2], different_inputs=True)
    info.split, info2.split = "test", "test"
    align.build_graph(id="two_models_different_real_inputs")

    # align = MultiAlignment([info, info_copy], different_inputs=False)
    # align.set_random(False, True)
    # info.split, info_copy.split = "train", "train"
    # align.build_graph(id="same_model_real_random_inputs")
    # info.split, info_copy.split = "test", "test"
    # align.build_graph(id="same_model_real_random_inputs")


if __name__ == '__main__':
    all_inter_experiments() #enable=["cub200", "code"]
    # real_and_unreal()
    # multi()
