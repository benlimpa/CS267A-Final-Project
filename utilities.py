from ogb.nodeproppred import NodePropPredDataset
import pandas as pd
import gzip


def get_nodeid2text():
    # load dataset
    dataset = NodePropPredDataset(name="ogbn-arxiv", root="./arxiv/")
    graph, label = dataset[0]  # graph: library-agnostic graph object

    # get splits
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )

    # extract labels
    nodelabels = pd.Series(label.flatten(), name="label")

    # extract and load mapping
    with gzip.open("arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz", "rb") as rf:
        with open("arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv", "wb") as wf:
            wf.write(rf.read())
    nodeid2paperid = pd.read_csv(
        "arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv", dtype=str
    )

    # join node ids with subject labels
    nodeid2paperid2label = pd.merge(
        nodeid2paperid, nodelabels, left_index=True, right_index=True
    )

    # load raw title and abstracts
    paperid2text = pd.read_csv(
        "arxiv/titleabs.tsv",
        sep="\t",
        dtype=str,
        names=["paper id", "title", "abstract"],
    )

    # join paper ids with title+abstracts
    nodeid2text = pd.merge(nodeid2paperid2label, paperid2text, on="paper id")

    # concatenate titles with abstracts
    nodeid2text = nodeid2text.assign(
        text=nodeid2text["title"].str.cat(nodeid2text["abstract"], sep=" ")
    )

    # project only the required columns
    nodeid2text = nodeid2text[["node idx", "label", "text"]]

    # split the dataset
    nodeid2text_train = nodeid2text.loc[train_idx]
    nodeid2text_valid = nodeid2text.loc[valid_idx]
    nodeid2text_test = nodeid2text.loc[test_idx]
    return nodeid2text_train, nodeid2text_valid, nodeid2text_test
