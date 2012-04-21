import numpy as np
import time

class KDTree:

    # adapted from wikipedia, http://en.wikipedia.org/wiki/K-d_tree
    def __init__(self, X, depth=0, alpha=None, split_median=False):

        if len(X.shape) != 2 or X.shape[0] ==0:
            return None

        (n,k) = X.shape
        self.n = n
        self.bounding_box = np.vstack([np.min(X), np.max(X)])
        self.alpha_sum = np.sum(alpha)

        # Split along the axis with the largest width
        #axis_widths = self.bounding_box[1,:] - self.bounding_box[0,:]
        #axis = np.argmin(axis_width)
        #self.axis = axis

        axis = depth % k
        self.axis = axis

        # Sort point list and choose median as pivot element
        X = X[X[:,axis].argsort(), :]
        median = n // 2 # choose median

        # Create node and construct subtrees
        self.location = X[median, :]
        self.left_child = KDTree(X[:median, :], depth + 1) if n > 1 else None
        self.right_child = KDTree(X[median + 1:, :], depth + 1) if n > 2 else None

    def insert(self, x):
        pass

    def nn(self, x):
        x = np.array(x)
        return self._nn_nocheck(x.reshape((-1,)))

    # nearest-neighbor lookup
    def _nn_nocheck(self, x):

        try:
            axis = self.axis
        except AttributeError:
            import pdb
            pdb.set_trace()

        v = x - self.location
        self_dist = np.dot(v,v)

        # if we're at a leaf, return the current point
        if self.left_child is None:
            return self.location, self_dist

        if x[axis] < self.location[axis] or self.right_child is None:
            other_child = self.right_child
            nn, nndist = self.left_child._nn_nocheck(x)
        else:
            other_child = self.left_child
            nn, nndist = self.right_child._nn_nocheck(x)

        # compare this current node to the the best point from further down the tree
        if self_dist < nndist:
            nndist = self_dist
            nn = self.location

        # possibly check the other side of the splitting plane
        split_dist = (x[axis] - self.location[axis])**2
        if split_dist < nndist and self.right_child is not None:
            nn2, dist2 = other_child._nn_nocheck(x)
            if dist2 < nndist:
               nndist = dist2
               nn = nn2

        return nn, nndist


def naive_nn(data, x):

    best_dist = np.float("inf")
    nn = None

    for d in data:
        v = x-d
        dist = np.dot(v,v)
        if dist < best_dist:
            best_dist = dist
            nn = d

    return nn, best_dist

def main():

    data = np.random.randn(10000, 2)
    t1 = time.time()
    tree = KDTree(data)
    t2 = time.time()
    print "constructing KDTree on 10000 points took %f seconds." % (t2-t1)


    testdata = np.random.randn(200, 2)

    treeNNs = []
    naiveNNs = []

    t1 = time.time()
    for testx in testdata:
        nn = tree.nn(testx)
        treeNNs.append(nn)
    t2 = time.time()
    print "doing 200 NN lookups with KDTree took %f seconds." % (t2-t1)

    for testx in testdata:
        nn = naive_nn(data, testx)
        naiveNNs.append(nn)
    t3 = time.time()
    print "doing 200 NN lookups naively took %f seconds." % (t3-t2)

    for i in range(200):
        if np.abs(treeNNs[i][1] - naiveNNs[i][1]) > 0.00001:
            print i, treeNNs[i], naiveNNs[i]
            import pdb
            pdb.set_trace()
    print "verified both methods returned identical NNs!"

if __name__ == "__main__":
    main()
