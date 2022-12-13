import sys
import numpy as np

# Calculate manhanttan distance
def man_distance(p, q):
    return np.sum(np.absolute(p - q), axis=1)

def euc_distance(p, q):
    ret = np.sqrt(np.sum((p - q)**2, axis=1))
    ret = ret.reshape(ret.shape[0],1)
    return ret

def get_centroids(dataset, cats):
    centroid = []
    for i in range(1, k+1):
        centroid.append(np.mean(dataset[cats[:, 0] == str(i)][:, :-1].astype('float'), axis=0).tolist())
    centroid = np.array(centroid)
    return centroid


if len(sys.argv) < 4:
    print("Usage: python3 k_means_cluster.py <data_file> <k> <iterations> ")
    exit()
file = 'UCI_datasets/' + sys.argv[1]
k = int(sys.argv[2])
iterations = int(sys.argv[3])

lines = []
with open(file) as f:
    for line in f.readlines():
        lines.append(line.split())
dataset = np.array(lines)
rand_cat = np.random.randint(1, k+1, size=(dataset.shape[0], 1)).astype('str')
centroids = get_centroids(dataset, rand_cat)


for i in range(iterations):
    distances = None
    for j in range(k):
        if distances is None:
            distances = euc_distance(dataset[:, :-1].astype('float'), centroids[j].reshape(1, dataset[:, :-1].shape[1]))
        else:
            distances = np.concatenate([distances,
                                        euc_distance(dataset[:, :-1].astype('float'), centroids[j].reshape(1, dataset[:, :-1].shape[1]))], axis=1)
    error = np.sum(np.sum(distances, axis=1), axis=0)
    if (i == 0):
        print("After initialization: error = {:.4f}".format(error))
    else:
        print("After iteration {}: error = {:.4f}".format(i+1, error))
    cat = (np.argmin(distances, axis=1).reshape(distances.shape[0], 1) + 1).astype('str')
    centroids = get_centroids(dataset, cat)
