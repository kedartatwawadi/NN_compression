import json
import pickle
import numpy as np
import matplotlib.pylab as plt

#Open the info file
def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

output_file = "output_0entropy_2.txt"
items = load(output_file)
epochs_per_k = {}
max_k=50

for k in range(1,max_k):
	#print k
	_epochs = []
	items = load(output_file)
	for item in items:
		#print item
		markovity = item['markovity']
		#print k,markovity
		if markovity==k:
			#print k
			_epochs.append(item['epoch_stopped'])

	if len(_epochs):
		_array = np.asarray(_epochs)
		#print _array
		epochs_per_k[k] = np.mean(_array)

# Plotting
#print epochs_per_k
epoch_stopped = sorted(epochs_per_k.items())
x, y = zip(*epoch_stopped)
plt.plot(x, y)
plt.show()


    


