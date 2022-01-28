# make a bunch of mongodb queries here
import pandas as pd

from array_graph import ArrayGraph
from align import get_best_timestamp, Alignment
from train_models import InfoMan, get_configs, get_parser
from helpers import validate_graph, change_names
from logger import MasterLog


class Conn:
    def __init__(self):
        self.db = None


def precision_recall_edge_stats():
    cols = {"dataset": [],
            "alpha": []}

    plot = {"dataset": [],
            "split": [],
            "precision": [],
            "recall": [],
            "paired": []}
    for dataset, args in get_configs().items():
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
        info.split = "train"
        align = Alignment(info)
        graph = align.load_graph()
        graph: ArrayGraph

        info.split = "test"
        align = Alignment(info)
        test_graph = align.load_graph()
        test_graph: ArrayGraph

        cols["dataset"].append(dataset)
        cols["alpha"].append(graph.alpha)

        for idx, e in enumerate(graph.g.edges):
            # train
            dat = graph.g[e[0]][e[1]]
            plot["dataset"].append(dataset)
            plot["precision"].append(dat["precision"])
            plot["recall"].append(dat["recall"])
            plot["split"].append("train")
            plot["paired"].append(idx)

            assert dat["precision"] == graph.get_precision(e[0], e[1])
            assert dat["recall"] == graph.get_recall(e[0], e[1])
            # test
            test_precision = test_graph.get_precision(e[0], e[1])
            test_recall = test_graph.get_recall(e[0], e[1])
            plot["dataset"].append(dataset)
            plot["precision"].append(test_precision)
            plot["recall"].append(test_recall)
            plot["split"].append("test")
            plot["paired"].append(idx)

    alpha_df = pd.DataFrame(cols)
    confu_df = pd.DataFrame(plot)
    alpha_df.to_csv("plots/alpha.csv")
    confu_df.to_csv("plots/confu.csv")


def alignment_stats(uni=False, intervention_type=0, version="alignment_results7"):
    master = MasterLog("general", version)
    col = master.collection
    train_best = {}
    valid_best = {}
    df = {"dataset": [],
          }
    for vali in (False, "test"):
        best = valid_best if vali else train_best
        for dataset in get_configs():
            keys = {"prepend": "alignment_stats",
                    "dataset": dataset,
                    "validation": vali,
                    "contradictions": {"$exists": 1},
                    "type": intervention_type}
            if uni:
                keys["unidirectional"] = uni
            res = col.find(keys)
            # make sure to change the version "alignment_results7" to your version so you don't load incorrectly
            res = res.sort("counterfactual_aligned", -1)
            r = next(iter(res))
            best[dataset] = r
        vali = True if vali == "test" else False
        #
        df.update({("aligned", vali): [],
                   ("unchanged", vali): [],
                   ("changed unaligned", vali): [],
                   ("original correct", vali): [],
                   ("contradictions", vali): [], })

        for dataset, docu in best.items():
            oc = docu["original_correct"]
            if vali:
                df["dataset"].append(dataset)
            df[("aligned", vali)].append(docu["counterfactual_aligned"] / oc)
            df[("unchanged", vali)].append(docu["counterfactual_unchanged"] / oc)
            df[("changed unaligned", vali)].append(docu["counterfactual_changed_unaligned"] / oc)
            df[("original correct", vali)].append(oc)
            df[("contradictions", vali)].append(docu["contradictions"])

    reord = {"dataset": df["dataset"]}
    for col in df:
        if col[1] is False:
            reord[col] = df[col]
            trcol = (col[0], True)
            reord[trcol] = df[trcol]

    df = pd.DataFrame(reord)
    df = df.set_index('dataset')
    df = df.round(5)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Accuracy', 'Valid'])

    df.to_csv(f"plots/align{'_uni' if uni else ''}.csv")
    print(df.to_latex())
    return df


def interleaving_stats():
    best = {}
    splits = ["train", "valid", "test"]
    df = {}
    for dataset in get_configs():
        master = MasterLog(dataset, "convergence_push")
        col = master.collection
        df[dataset] = {}

        res = col.find({"prepend": "initial_avgs",
                        "split": "test"})
        res = res.sort("original correct", -1)
        ini = next(iter(res))

        res = col.find({"split": "test",
                        "prepend": "final"})
        res = res.sort("original correct", -1)
        fin = next(iter(res))

        df[dataset] = (ini, fin)
    # process
    processed = {}
    for ds, splits in df.items():
        processed[ds] = {}
        for split, vals in splits.items():
            processed[ds][split] = (vals[0]['original_correct'], vals[1]['original_correct'])

    tests = {}
    for ds, splits in df.items():
        tests[ds] = processed[ds]['test']
    mul = {"dataset": [],
           "before": [],
           "after": []}
    for ds in tests:
        mul["dataset"].append(ds)
        bef, aft = tests[ds]
        mul["before"].append(bef)
        mul["after"].append(aft)

    df = pd.DataFrame(mul)
    df.to_csv("plots/interleave.csv")
    print(df.to_latex())


def inter_layer_analysis():
    configs = get_configs()
    args = configs["mnist"]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.args["inter_layer"] = True
    info.load_checkpoint()
    align = Alignment(info)
    graph = align.load_graph()
    pass

    info.split = "valid"
    valid_graph = align.load_graph()
    precisions, edges, train_acc, valid_acc = validate_graph(graph, valid_graph)
    print(f"{train_acc=}, {valid_acc=}")

    for e in graph.g.edges:
        a, b = e
        ranges = [10, 138, 266]
        if 10 <= a[1] < 138 and 138 < b[1] < 266:
            print(f"{e}:{graph.g[a][b]}. Train/valid acc: {precisions[e]}")


def alignment_table():
    df = alignment_stats(True, 0, "alignment_results7")
    # df = alignment_stats(True, 2, "rerun2")
    # df["beta"] = 1
    # cols = df.columns.to_list()
    # cols = [cols[-1]]+cols[:-1]
    #
    # df2 = alignment_stats(False)
    # beta=[]
    # for dataset in df.index:
    #     configs = get_configs()[dataset]
    #     args = get_parser().parse_args(args=configs.split())
    #     beta.append(args.bidi_recall_thres)
    # df2["beta"] = beta
    #
    # combined = pandas.concat([df, df2])
    # combined = df
    # combined = combined[cols]
    df = change_names(df)

    for col in df:
        if col not in (('beta', ''), ('contradictions', False), ('contradictions', True)):
            df[col] = df[col] * 100
    print(df.to_latex(float_format="%.2f"))
    print("DONE")


if __name__ == '__main__':
    precision_recall_edge_stats()
    alignment_table()
    # inter_graph_analysis()
