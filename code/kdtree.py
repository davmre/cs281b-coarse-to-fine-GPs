import numpy as np
import time

class KDTree:

    def __init__(self, X, depth=0):

        if len(X.shape) != 2 or X.shape[0] ==0:
            return None

        (n,k) = X.shape
        self.n = n
        self.k = k
        self.depth = depth

        # if this is a leaf node, just store the appropriate point
        if n == 1:
            self.x = X[0,:]
            self.left_child = None
            self.right_child = None
            return

        # otherwise, create two child nodes, splitting along the axis
        # with the largest width.
        self.bounding_box = np.vstack([np.min(X, axis=0), np.max(X, axis=0)])

        axis_widths = self.bounding_box[1,:] - self.bounding_box[0,:]
        self.axis = np.argmax(axis_widths)
        self.split_value = axis_widths[self.axis]/2 + self.bounding_box[0, self.axis]

        left_points = (X[:, self.axis] <= self.split_value)
        right_points = np.invert(left_points)
        self.left_child = KDTree(X[left_points, :], depth + 1)
        self.right_child = KDTree(X[right_points, :], depth + 1)

    def insert(self, x):
        pass

    def update_p(self, X,p):

        self.p_sum = np.sum(p)

        if self.left_child is not None:
            left_points = (X[:, self.axis] <= self.split_value)
            right_points = np.invert(left_points)
            self.left_child.update_p(X[left_points,:], p[left_points])
            self.right_child.update_p(X[right_points,:], p[right_points])

    def exact_weighted_sum(self, x, kernel):
        if self.left_child is None:
            return self.p_sum * kernel(x, self.x), self.p_sum

        else:
            left_sum, left_w = self.left_child.exact_weighted_sum(x, kernel)
            right_sum, right_w = self.right_child.exact_weighted_sum(x, kernel)
            return left_sum + right_sum, left_w + right_w

    def weighted_sum(self, x, kernel, wSoFar, epsilon):
        if self.left_child is None:
            return self.p_sum * kernel(x, self.x), wSoFar+self.p_sum


        # if our query point is not in the bounding box, compute
        # weight bounds using the closest and nearest corners of the
        # bounding box.
        if self.n >2 and ((x - self.bounding_box[0,:]) < 0 ).any() or ((x - self.bounding_box[1,:]) > 0 ).any():
            close_pt = np.copy(self.bounding_box[0,:])
            far_pt = np.copy(self.bounding_box[1,:])

            for i in range(self.k):
                c1 = self.bounding_box[0,i]
                c2 = self.bounding_box[1,i]
                if np.abs(x[i] - c1) > np.abs(x[i] - c2):
                    close_pt[i] = c2
                    far_pt[i] = c1
                wmin = kernel(x, far_pt)
                wmax = kernel(x, close_pt)

            # stop if the difference in weights is less than some
            # fraction of a lower bound on the average weight
            if self.n*(wmax-wmin) < 2*epsilon*(wSoFar + self.n*wmin):
                s = .5 * (wmax + wmin) * self.p_sum
                wSoFar += self.n * wmin
                return s, wSoFar

        if x[self.axis] <= self.split_value:
            near_child = self.left_child
            far_child = self.right_child
        else:
            near_child = self.right_child
            far_child = self.left_child

        s_nearer, wSoFar = near_child.weighted_sum(x, kernel, wSoFar, epsilon)
        s_farther, wSoFar = far_child.weighted_sum(x, kernel, wSoFar, epsilon)
        s = s_nearer + s_farther

        return s, wSoFar


    def nn(self, x):
        x = np.array(x)
        return self._nn_nocheck(x.reshape((-1,)))

    # nearest-neighbor lookup
    def _nn_nocheck(self, x):

        # if we're at a leaf, return the current point
        if self.left_child is None:
            v = x - self.x
            self_dist = np.dot(v,v)
            return self.x, self_dist

        # otherwise, check both subtrees
        axis = self.axis
        if x[axis] <= self.split_value:
            other_child = self.right_child
            nn, nndist = self.left_child._nn_nocheck(x)
        else:
            other_child = self.left_child
            nn, nndist = self.right_child._nn_nocheck(x)

        # possibly check the other side of the splitting plane
        split_dist = (x[axis] - self.split_value)**2
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
