curl https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz | gzip -d | sed "s/tsv.*200971/tsv\r200971/" > arxiv/titleabs.tsv
