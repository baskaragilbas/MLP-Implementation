import numpy as np
from module import MLP
from pathlib import Path

path = Path(__file__).parents[0]
file = str(path) + "\\Iris3.csv"

data = np.genfromtxt(file, skip_header=True, delimiter=',')

mlp1 = MLP(data,1000,0.8,80) #data, epoch, k-fold, learning rate 0.1

#uncomment one only
mlp1.run()
