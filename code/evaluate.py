import numpy as np
import scipy.io
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import csv
import munge
import gp_regression as gpr
from kernels import Gamma, InvGamma, LogNormal

import os,time

def process_data():
    convertfunc = lambda s: float(0.0) if s=='M' else float(1.0) if s=='F' else float(2.0) if s=='I' else float(-1)
    data = np.genfromtxt("abalone.data", delimiter=',', converters={0:convertfunc})
    data = data.view(float).reshape((data.shape[0], -1))
    idata, d = munge.categorical_to_indicators(data, 0)

    X, y, nX, ny = munge.preprocess(idata, target=(idata.shape[1]-1))
    np.savetxt("raw_X.dat", X)
    np.savetxt("raw_y.dat", y)
    np.savetxt("cooked_X.dat", nX)
    np.savetxt("cooked_y.dat", ny)


def load_seismic_station(siteid, phaseid=0, folder="../data/"):
    reader = csv.reader(open(os.path.join(folder, 'tt_data.csv'), 'rb'), delimiter=',')
    data = np.array([[float(col) for col in row] for row in reader if int(row[0])==int(siteid)] and int(row[1])==int(phaseid))
    data = munge.shuffle_data(data)

    train_n = .8 * data.shape[0]
    validate_n = .1 * data.shape[0]

    # cols: [siteid, phaseid, evlon, evlat, evdepth, sitelon, sitelat, siteheight, tt, ttres]
    X = data[0:train_n, [2,3,4,5,6,7] ]
    y = data[0:train_n, 9]

    v_cutoff = train_n+validate_n
    validation_X = data[train_n:v_cutoff, [2,3,4,5,6,7] ]
    validation_y = data[train_n:v_cutoff, 9]
    test_X = data[v_cutoff:, [2,3,4,5,6,7] ]
    test_y = data[v_cutoff:, 9]

    return X, y, validation_X, validation_y, test_X, test_y

def load_yearpred(folder="../data/"):
    data = np.genfromtxt(os.path.join(folder, 'YearPredictionMSD.txt'), delimiter=',')

    test_X = data[463715:, 1:]
    test_y = data[463715:, 0]

    validation_X = data[410000:463715, 1:]
    validation_y = data[410000:463715, 0]

    train = munge.shuffle_data(data[0:410000, :])
    train_X = train[:, 1:]
    train_y = train[:, 0]

    return train_X, train_y, validation_X, validation_y, test_X, test_y

def load_sarcos(folder="../data/"):
    sarcos_inv = scipy.io.loadmat(os.path.join(folder, "sarcos_inv.mat"))['sarcos_inv']
    sarcos_inv_test = scipy.io.loadmat(os.path.join(folder, "sarcos_inv_test.mat"))['sarcos_inv_test']

    X = sarcos_inv[0:40000, 0:21]
    y = sarcos_inv[0:40000, 22]
    validation_X = sarcos_inv[40000:, 0:21]
    validation_y = sarcos_inv[40000:, 22]
    test_X = sarcos_inv_test[:, 0:21]
    test_y = sarcos_inv_test[:, 22]

    return X, y, validation_X, validation_y, test_X, test_y


def plot_learning_curve(X, y, validation_X, validation_y, sizes=(100, 300, 500, 1000, 3000, 5000, 8000, 10000)):

	widths = munge.distance_quantiles(X, [0.5])
	print "widths", widths

	sizes = np.array(sizes).reshape(-1,1)
	padding = np.zeros((len(sizes), 4))
	results = np.hstack([sizes, padding])

	for i,n in enumerate(sizes):
		limX = X[0:n, :]
		limy = y[0:n]
		t1 = time.time()
		gp = gpr.GaussianProcess(X = limX, y = limy, kernel="se_iso", kernel_params=np.asarray((1, 1, widths[0])))
		t2 = time.time()
		results[i, 1] = t2-t1
		py = gp.predict(validation_X)
		t3 = time.time()
		results[i, 2] = t3-t2
		results[i, 3] = gpr.rms_loss(py,validation_y)
		results[i, 4] = gpr.abs_loss(py,validation_y)

		print "done with n =",n
		print "got row", results[i,:]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot ( results[:,0],results[:,1],'b-',label='train time' )
	ax1.plot ( results[:,0],results[:,2],'g-',label='test time' )
	ax1.set_xlabel("n")
	ax1.set_ylabel("seconds", color='b')

	ax2 = ax1.twinx()
	ax2.plot ( results[:,0],results[:,3],'r-',label='loss' )
	ax2.set_ylabel("root mean squared loss")

	plt.savefig("curve.pdf")

def main():

	X, y, validation_X, validation_y, test_X, test_y = load_sarcos()
	plot_learning_curve(X,y,validation_X,validation_y)

if __name__ == "__main__":
	main()

