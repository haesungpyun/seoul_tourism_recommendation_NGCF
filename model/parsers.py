import argparse

parser = argparse.ArgumentParser(description='Run NGCF')
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch_size',type=int,default=256,help="Batch size")
parser.add_argument('-bt','--test_batch',type=int,default=10,help="Test Batch size")
parser.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer')
parser.add_argument('-k','--ks',type=int,default=10,help='choose top@k for NDCG@k, HR@k')
parser.add_argument('-emb','--embed_size',type=int,default=64,help='choose embedding size')
parser.add_argument('-mlp','--mlp_ratio',type=float,default=0.5,help='choose ration between embedding and mlp weight from user features')
parser.add_argument('-n','--node_dropout',type=float,default=0.2,help='choose node dropout ratio')
parser.add_argument('-m','--mess_dropout',type=list,default=[0.1,0.1,0.1],help='choose message dropout ratio')
args = parser.parse_args()
