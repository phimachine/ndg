# table 2
from pathlib import Path

import networkx
import pandas

from align import *
from helpers import validate_graph, change_names


def build_trivials():
    # build_all_graphs()
    build_all_graphs("--trivial_random_dataset True")
    build_all_graphs("--trivial_random_model True")


def rebuild_test(enable=None):
    configs = get_configs()
    for arg_str in ("--trivial_random_dataset True", "--trivial_random_model True"):
        for ds, args in configs.items():
            if enable and ds not in enable:
                continue
            else:
                align(args + " " + arg_str + " ", False, uni=False, build_train=False, build_valid=False, build_test=True,
                      interchange_intervention=False, reuse_counts=False)


def table2(enable=None):
    configs = get_configs()
    cols = {"dataset": [],
            "neurons": [],
            "nodes": [],
            "edges": [],
            "isolated": [],
            "equivalent": [],
            "constant": [],
            "degree": [],
            "height": [],
            "cycles": [],
            "contradictions": [],
            "components": []}
    for ds, args in configs.items():
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
        info.split = "train"
        # info.load_checkpoint()
        align = Alignment(info)
        graph = align.load_graph()
        cols['dataset'].append(ds)
        ret = table2_stats(graph)
        for k, v in ret.items():
            cols[k].append(v)
    df = pandas.DataFrame(cols)
    df = df.set_index("dataset")
    path = save_dir / "figures" / "table2.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        df.to_csv(f)
    df = change_names(df)
    print(df.to_latex(float_format="%.1f"))

    pass


def table2_stats(graph):
    ret = {}

    # neurons
    ret["neurons"] = graph.num_latent_predicates

    # nodes
    ret["nodes"] = len(graph.g.nodes)

    # edges
    ret["edges"] = len(graph.g.edges)

    # isolated
    cnt = 0
    for node in graph.g.nodes:
        pred = graph.g.predecessors(node)
        if next(pred, None) is None:
            succ = graph.g.successors(node)
            if next(succ, None) is None:
                cnt += 1
    ret["isolated"] = cnt

    # equivalent
    try:
        equiv = len(set(graph.smallest_equiv.keys()).union(graph.smallest_equiv.values())) * 2
    except AttributeError:
        equiv = len(set(graph._equivalence.keys()).union(graph._equivalence.values())) * 2

    ret["equivalent"] = equiv

    # constant
    ret["constant"] = len(graph.constants)

    # degree
    total_in = 0
    total_out = 0
    for node in graph.g.nodes:
        total_in += len(list(graph.g.predecessors(node)))
        total_out += len(list(graph.g.successors(node)))

    ret["degree"] = total_in / len(graph.g.nodes)

    # height
    try:
        h = networkx.dag.dag_longest_path_length(graph.g)
    except networkx.NetworkXUnfeasible:
        raise
    ret["height"] = h

    # cycles
    ret["cycles"] = len(list(networkx.simple_cycles(graph.g)))

    # contradiction
    ret["contradictions"] = 0
    for node in graph.g.nodes:
        descs = set(networkx.descendants(graph.g, node))
        for d in descs:
            if (not d[0], d[1]) in descs:
                ret["contradictions"] += 1
                break

    # components
    ret["components"] = networkx.number_weakly_connected_components(graph.g)

    return ret


def mnist_ndg_figure_2():
    configs = get_configs()
    cols = {"dataset": [],
            "neurons": [],
            "nodes": [],
            "edges": [],
            "height": []}
    ds = "mnist"
    args = configs[ds]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()
    align = Alignment(info)
    graph = align.load_graph()
    dot_path = save_dir / "figures" / "figure2.dot"
    dot, g = graph.nice_plot(graph.g, dot_path, save='raw')
    for node in dot.get_nodes():
        node.set_height(1)
        node.set_width(1)
        node.set_fontsize(30)
    dot.set_ratio("0.1")
    # dot.set_size("8.3,11.7!")
    dot.write(save_dir / "figures" / "figure2.eps", format="eps")

def allnli_figure():
    configs = get_configs()
    cols = {"dataset": [],
            "neurons": [],
            "nodes": [],
            "edges": [],
            "height": []}
    ds = "allnli"
    args = configs[ds]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()
    align = Alignment(info)
    graph = align.build_graph()
    dot_path = save_dir / "figures" / "allnli.dot"
    dot, g = graph.nice_plot(graph.g, dot_path, save='raw')
    for node in dot.get_nodes():
        node.set_height(1)
        node.set_width(1)
        node.set_fontsize(30)
    # dot.set_ratio("0.1")
    # dot.set_size("8.3,11.7!")
    dot.write(save_dir / "figures" / "allnli.eps", format="eps")


def inspect(graph):
    edges = list(graph.g.edges)
    random.shuffle(edges)
    for idx, e in enumerate(edges):
        print(f"recall = {graph.g[e[0]][e[1]]['recall']},  precision = {graph.g[e[0]][e[1]]['precision']}")
        if idx == 100:
            break


def accuracy_cloud(trivial_random_model=False, trivial_random_dataset=False):
    df = {"train": [],
          "test": [],
          "dataset": []}
    df_stat = {"dataset": [],
               "edges": [],
               "train_acc": [],
               "test_acc": []}

    configs = get_configs()
    for ds in configs:
        # if ds != "cub200":
        #     continue
        args = configs[ds]
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp)
        if trivial_random_model:
            info.trivial_random_model = True
        if trivial_random_dataset:
            info.trivial_random_dataset = True
        info.load_checkpoint()
        info.split = "train"
        align = Alignment(info)
        align.info.split = "train"
        train_graph = align.load_graph()
        align.info.split = "test"
        test_graph = align.load_graph()

        precisions, num_edges, train_acc, test_acc = validate_graph(train_graph, test_graph)
        for e in precisions:
            train, test = precisions[e]
            df["train"].append(train)
            df["test"].append(test)
            df["dataset"].append(ds)
        df_stat["dataset"].append(ds)
        df_stat["edges"].append(num_edges)
        df_stat["train_acc"].append(train_acc)
        df_stat["test_acc"].append(test_acc)

    df = pandas.DataFrame(df)
    name = "cloud.csv"
    if trivial_random_model:
        name = "trivial_random_model_" + name
    if trivial_random_dataset:
        name = "trivial_random_dataset_" + name
    df.to_csv(Path("plots") / name)
    return df, df_stat


def precisions_to_box_plot(precisions):
    pass


def sanity():
    # table 3
    df = {"dataset": [],
          "original_edges": [],
          "original_train_acc": [],
          "original_test_acc": [],
          "random_model_edges": [],
          "random_model_train_acc": [],
          "random_model_test_acc": [],
          "random_data_edges": [],
          "random_data_train_acc": [],
          "random_data_test_acc": [],
          }
    original_edges, original = accuracy_cloud(False, False)
    random_model_edges, random_model = accuracy_cloud(True, False)
    random_data_edges, random_data = accuracy_cloud(False, True)
    pass
    df["dataset"] = original["dataset"]
    keys = list(df.keys())
    key_idx = 1
    for ret in (original, random_model, random_data):
        for key, val in ret.items():
            if key == "dataset":
                continue
            if "acc" in key:
                val = [v * 100 for v in val]
            df[keys[key_idx]] = val
            key_idx += 1
    df = pandas.DataFrame(df)
    df = df.set_index("dataset")
    df = change_names(df)
    print(df.to_latex(float_format="%.2f"))
    # df2 = {"dataset": [],
    #        "original_edges": [],
    #        "original_acc": [],
    #        "random_model_edges": [],
    #        "random_model_acc": [],
    #        "random_data_edges": [],
    #        "random_data_acc": []
    #        }
    # for ds in df.index:
    #     row = df.loc[ds]
    #     df2["dataset"].append(ds)
    #     df2["original_edges"].append(row["original_edges"])
    #     df2["original_acc"].append(f"{row['original_train_acc']:.2f} / {row['original_test_acc']:.2f}")
    #     df2["random_model_edges"].append(row["random_model_edges"])
    #     df2["random_model_acc"].append(f"{row['random_model_train_acc']:.2f} / {row['random_model_test_acc']:.2f}")
    #     df2["random_data_edges"].append(row["random_data_edges"])
    #     df2["random_data_acc"].append(f"{row['random_data_train_acc']:.2f} / {row['random_data_test_acc']:.2f}")
    # df2 = pandas.DataFrame(df2)
    # df2 = df2.set_index("dataset")
    # df2["original_edges"] = df2["original_edges"].astype(int)
    # df2["random_model_edges"] = df2["random_model_edges"].astype(int)
    # df2["random_data_edges"] = df2["random_data_edges"].astype(int)
    #
    # print(df2.to_latex())
    pass


def cross_acc():
    # table 7
    """
        Real graph edges on random inputs
        Random input graph with real inputs
        Do they hold?
        Hope not.
    """
    df = {}
    configs = get_configs()
    for ds in configs:
        # if ds != "cub200":
        #     continue
        args = configs[ds]
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp)
        info.load_checkpoint()
        info.split = "train"
        real_align = Alignment(info)
        real_graph = real_align.load_graph()

        info = InfoMan(args, timestamp=best_timestamp)
        info.trivial_random_dataset = True
        info.load_checkpoint()
        info.split = "train"
        random_align = Alignment(info)
        random_graph = random_align.load_graph()

        real_graph_random_data = validate_graph(real_graph, random_graph)
        random_graph_real_data = validate_graph(random_graph, real_graph)
        df[ds] = (real_graph_random_data, random_graph_real_data)
    table = {"dataset": [],
             "G real": [],
             "G random": [],
             "G' real": [],
             "G' random": []
             }
    for ds in df:
        table["dataset"].append(ds)
        table["G real"].append(df[ds][0][2])
        table["G random"].append(df[ds][0][3])
        table["G' real"].append(df[ds][1][3])
        table["G' random"].append(df[ds][1][2])
    table = pandas.DataFrame(table)
    table = table.set_index("dataset")
    table = change_names(table)

    for col in table:
        table[col] = table[col] * 100
    print(table.to_latex(float_format="%.2f"))
    return table


def a():
    es = real_graph.g.edges
    for e in es:
        a, b = e
        random_graph.implies(a, b)
        break


def ribbon_plot():
    """
    For every dataset, relationship between quantiles and alpha
    """
    df = {"dataset": [],
          "alpha": [],
          "q1": [],
          "q2": [],
          "q3": [],
          "average": [],
          "split": [],
          "edges": [],
          "all_edges": []
          }
    configs = get_configs()
    test_graphs = {}

    for alpha in alpha_range():
        # complete = True
        # rows = len(df["dataset"])

        for ds in configs:
            if ds not in test_graphs:
                args = configs[ds]
                args = get_parser().parse_args(args.split())
                args.shuffle = False
                best_timestamp = get_best_timestamp(args)
                info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
                info.split = "test"
                align = Alignment(info)
                test_graph = align.load_graph()
                test_graphs[ds] = test_graph
            else:
                test_graph = test_graphs[ds]

            args = configs[ds]
            args += f" --alpha {alpha} "
            args = get_parser().parse_args(args.split())
            args.shuffle = False
            best_timestamp = get_best_timestamp(args)
            info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
            info.split = "train"
            align = Alignment(info)
            # try:
            ag = align.load_graph()
            # except FileNotFoundError:
            #     complete = False
            #     break

            prec, _, _, _ = validate_graph(ag, test_graph)
            for im, train in enumerate(("train", "test")):
                accs = [prec[p][im] for p in prec]
                accs = [a for a in accs if a is not None]
                try:
                    q1, q2, q3 = quantile(accs)
                    df["average"].append(sum(accs) / len(accs))
                except IndexError:
                    q1, q2, q3 = np.nan, np.nan, np.nan
                    df["average"].append(np.nan)

                df["dataset"].append(ds)
                df["alpha"].append(alpha)
                df["q1"].append(q1)
                df["q2"].append(q2)
                df["q3"].append(q3)
                df["split"].append(train)
                df["edges"].append(len(ag.g.edges))
                all_edges = len(ag.g.edges)
                for n in ag.g.nodes:
                    all_edges += len(ag.get_all_equiv(n))
                df["all_edges"].append(all_edges)

        # if only_complete and not complete:
        #     for col in df:
        #         df[col] = df[col][:rows]

    df = pandas.DataFrame(df)
    df.set_index("dataset")
    df = change_names(df)
    path = save_dir / "figures" / "alpha_ribbon.csv"
    with path.open('w') as f:
        df.to_csv(f)
    return df


def quantiles_acc(array_graph):
    ag = array_graph
    accs = []
    for e in ag.g.edges:
        a, b = e
        accs.append(ag.g[a][b]["precision"])
    accs = sorted(accs)
    l = len(accs)
    q = l // 4
    q1, q2, q3 = accs[q], accs[2 * q], accs[3 * q]
    return q1, q2, q3


def quantile(lst):
    accs = sorted(lst)
    l = len(accs)
    q = l // 4
    q1, q2, q3 = accs[q], accs[2 * q], accs[3 * q]
    return q1, q2, q3


def inter_model_table(load=False):
    if load:
        path = save_dir / "figures" / "inter_model.csv"
        with open(path, "r") as f:
            df = pandas.read_csv(f)

        for a in df:
            if "acc" in a:
                df[a] *= 100
        df = df.set_index("dataset")
        print(df.to_latex(float_format="%.2f"))
        return df

    df = ["dataset", "inter_model_edges", "inter_model_training_acc", "inter_model_test_acc",
          "different_input_edges", "different_input_edges_train_acc", "different_input_edges_test_acc",
          "two_models_different_real_inputs_edges", "two_models_different_real_inputs_train_acc",
          "two_models_different_real_inputs_test_acc"]
    df = {d: [] for d in df}
    configs = get_configs()
    for dataset in configs:
        df["dataset"].append(dataset)
        configs = get_configs()
        args = configs[dataset]
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)

        info_copy = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)

        args = configs[dataset]
        args = get_parser().parse_args(args.split())
        args.secondary = True
        args.version = args.version + '_secondary'
        best_timestamp = get_best_timestamp(args)
        info2 = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)

        align = MultiAlignment([info, info2], different_inputs=False)
        info.split, info2.split = "train", "train"
        two_models_same_inputs_train_graph = align.load_graph(id="two_models_same_inputs")
        info.split, info2.split = "test", "test"
        two_models_same_inputs_test_graph = align.load_graph(id="two_models_same_inputs")
        prune_inter_connections(two_models_same_inputs_train_graph)
        _, edges, train_acc, test_acc = validate_graph(two_models_same_inputs_train_graph,
                                                       two_models_same_inputs_test_graph)
        df["inter_model_edges"].append(edges)
        df["inter_model_training_acc"].append(train_acc)
        df["inter_model_test_acc"].append(test_acc)

        align = MultiAlignment([info, info_copy], different_inputs=True)
        info.split, info_copy.split = "train", "train"
        same_model_different_real_inputs_train_graph = align.load_graph(id="same_model_different_real_inputs")
        info.split, info_copy.split = "test", "test"
        same_model_different_real_inputs_test_graph = align.load_graph(id="same_model_different_real_inputs")
        prune_inter_connections(same_model_different_real_inputs_train_graph)
        _, edges, train_acc, test_acc = validate_graph(same_model_different_real_inputs_train_graph,
                                                       same_model_different_real_inputs_test_graph)
        df["different_input_edges"].append(edges)
        df["different_input_edges_train_acc"].append(train_acc)
        df["different_input_edges_test_acc"].append(test_acc)

        align = MultiAlignment([info, info2], different_inputs=True)
        info.split, info_copy.split = "train", "train"
        same_model_real_random_inputs_train_graph = align.load_graph(id="two_models_different_real_inputs")
        info.split, info_copy.split = "test", "test"
        same_model_real_random_inputs_test_graph = align.load_graph(id="two_models_different_real_inputs")
        prune_inter_connections(same_model_real_random_inputs_train_graph)
        _, edges, train_acc, test_acc = validate_graph(same_model_real_random_inputs_train_graph,
                                                       same_model_real_random_inputs_test_graph)
        df["two_models_different_real_inputs_edges"].append(edges)
        df["two_models_different_real_inputs_train_acc"].append(train_acc)
        df["two_models_different_real_inputs_test_acc"].append(test_acc)

    del df["two_models_different_real_inputs_edges"], df["two_models_different_real_inputs_train_acc"], df[
        "two_models_different_real_inputs_test_acc"]

    df2 = pandas.DataFrame(df)
    df2 = df2.set_index("dataset")
    df2 = change_names(df2)
    path = save_dir / "figures" / "inter_model.csv"
    with open(path, "w") as f:
        df2.to_csv(f)

    print(df2)
    df2 = inter_model_table(True)
    return df2


def prune_inter_connections(array_graph):
    targets = array_graph.num_preds - array_graph.num_latent_predicates
    latent = array_graph.num_latent_predicates // 2
    remove_edges = []
    for e in array_graph.g.edges:
        a, b = e
        if targets <= a[1] < targets + latent and b[1] >= targets + latent:
            pass
        else:
            remove_edges.append(e)
    array_graph.g.remove_edges_from(remove_edges)


def inter_layer_table(load=False):
    """
    Takes an hour
    """
    if load:
        path = save_dir / "figures" / "inter_layer.csv"
        with open(path, "r") as f:
            df = pandas.read_csv(f)
        print(df.to_latex(float_format="%.2f"))

        return df

    df = ["dataset", "1_edges", "1_train", "1_test", "2_edges", "2_train", "2_test", "inter_edges",
          "inter_train", "inter_test"]
    df = {d: [] for d in df}

    configs = get_configs()
    shapes = inter_layer_num_preds()
    for dataset in configs:
        df["dataset"].append(dataset)
        configs = get_configs()
        args = configs[dataset]
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
        info.split = "train"
        info.args["inter_layer"] = True
        info.args["max_vector_input"] = 1e5
        align = Alignment(info)
        train_graph = align.load_graph()
        info.split = "test"
        test_graph = align.load_graph()

        edges_1 = []
        edges_2 = []
        edges_inter = []
        targets = train_graph.num_preds - train_graph.num_latent_predicates
        latent = shapes[dataset][0]
        for e in train_graph.g.edges:
            a, b = e
            if targets <= a[1] < targets + latent <= b[1]:
                edges_inter.append(e)
            elif targets <= a[1] < targets + latent and targets <= b[1] < targets + latent:
                edges_1.append(e)
            elif a[1] >= targets + latent and b[1] >= targets + latent:
                edges_2.append(e)

        # graph 1
        kept_graph = deepcopy(train_graph)
        kept_graph.g.remove_edges_from(edges_2 + edges_inter)
        _, edges, train_acc, test_acc = validate_graph(kept_graph, test_graph)
        df["1_edges"].append(edges)
        df["1_train"].append(train_acc)
        df["1_test"].append(test_acc)

        # graph 2
        kept_graph = deepcopy(train_graph)
        kept_graph.g.remove_edges_from(edges_1 + edges_inter)
        _, edges, train_acc, test_acc = validate_graph(kept_graph, test_graph)
        df["2_edges"].append(edges)
        df["2_train"].append(train_acc)
        df["2_test"].append(test_acc)

        # inter graph
        kept_graph = deepcopy(train_graph)
        kept_graph.g.remove_edges_from(edges_1 + edges_2)
        _, edges, train_acc, test_acc = validate_graph(kept_graph, test_graph)
        df["inter_edges"].append(edges)
        df["inter_train"].append(train_acc)
        df["inter_test"].append(test_acc)

    df = pandas.DataFrame(df)
    df = df.set_index("dataset")

    path = save_dir / "figures" / "inter_layer.csv"
    for a in df:
        if "edges" not in a and "dataset" not in a:
            df[a] *= 100

    df = change_names(df)

    with open(path, "w") as f:
        df.to_csv(f)

    print(df.to_latex(float_format="%.2f"))
    return df


def alpha_range():
    return [1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500, 1000, 5000, 100000]


def alpha_sweep(uni=True):
    configs = get_configs()
    pbar = tqdm(total=len(configs) * len(alpha_range()))

    for ds, args in configs.items():
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp)
        info.split = "train"
        align = Alignment(info)
        main_graph = align.load_graph()

        for alpha in alpha_range():
            args = configs[ds] + f" --alpha {alpha} "
            if args:
                args = get_parser().parse_args(args.split())
            else:
                args = get_parser().parse_args()
            args.shuffle = False
            if uni:
                args.bidi_recall_thres = 1
            best_timestamp = get_best_timestamp(args)
            info = InfoMan(args, timestamp=best_timestamp, no_model=True, no_dataset=True)
            info.split = "train"
            align = Alignment(info)
            if not align.get_graph_pickle_path().exists():
                align.build_graph(verbose=False, count_array=deepcopy(main_graph.count),
                                  num_latent=main_graph.num_latent_predicates, num_labels=main_graph.labels)
            pbar.update(1)
    pbar.close()


import traceback


def rerun():
    # try:
    #     rebuild_test()
    # except:
    #     traceback.print_exc()
    #     print(-1)
    #     return

    # try:
    #     inter_model_table(load=False)
    # except:
    #     traceback.print_exc()
    #     print(0)
    #     return
    # table2()

    # try:
    #     sanity()
    # except:
    #     traceback.print_exc()
    #     print(1)
    #     return

    # try:
    #     ribbon_plot()
    # except:
    #     traceback.print_exc()
    #     print(2)
    #     return
    #
    # try:
    #     # inter_layer("mnist")
    #     # inter_layer("mnist2")
    #     # inter_layer("sst2")
    #     # inter_layer("allnli")
    #     # inter_layer("cub200")
    #     # inter_layer("code")
    #
    #     inter_layer_table()
    # except:
    #     traceback.print_exc()
    #     print(3)
    #     return

    # ribbon_plot()
    cross_acc()

if __name__ == '__main__':
    rerun()
    # alpha_sweep()
