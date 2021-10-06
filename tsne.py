import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.manifold import TSNE


np.random.seed(42)

path = os.path.join(os.getcwd(), 'figures')
print('Current path is:', path)

if os.path.exists(path) == True:
    pass
    print('Path already existed.')
else:
    os.mkdir(path)
    print('Path created.')

x_values = pd.read_csv('./datasets/3_512_x_main.csv').values
y_values = pd.read_csv('./datasets/3_512_y_main.csv').values.ravel()
tsne_descriptors = TSNE(n_components=2, random_state = 42)
x_tsne = tsne_descriptors.fit_transform(x_values)


# draw tsne
cm = plt.cm.get_cmap('RdYlBu')
plt.figure(figsize=(16, 8))
plt.xticks(size = 22)
plt.yticks(size = 22)
plt.xlabel('t-SNE Reduced Dimension 1',fontproperties = 'Times New Roman', size = 24)
plt.ylabel('t-SNE Reduced Dimension 2',fontproperties = 'Times New Roman', size = 24)
plt.scatter(x_tsne[:, 0], x_tsne[:,1],c= y_values,vmin= 3, vmax= 9, s= 20, cmap=cm)
plt.colorbar()
plt.text(58, -70, "pChEMBL Value", fontproperties = 'Times New Roman', size = 18)
plt.savefig("figures/TSNE.png", bbox_inches='tight', pad_inches= 0)

tsne_image = Image.open('figures/TSNE.png')
tsne_image.show()
