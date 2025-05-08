# load the pkl file
import pickle

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    pkl_file = "/mnt/250T_ceph/tristanysui/okgqa/subgraphs/pruned_ppr_init/2.pkl"
    G = load_pkl(pkl_file)
    print(G)
    print(G.nodes(data=True))
    print(G.edges(data=True))
