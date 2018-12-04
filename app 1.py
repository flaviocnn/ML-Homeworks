import glob
from os import listdir
from os.path import join
from random import shuffle
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

DATA_FOLDER = "./PACS_homework/"

def fetchClasses():
    classes = [ folder for folder in listdir(DATA_FOLDER)]
    print(classes)
    return classes


def main():
    classes = fetchClasses()
    dataset = []

    for class_folder in classes:
        imgs = [join(DATA_FOLDER,join(class_folder,file)) for file in listdir(join(DATA_FOLDER,class_folder)) if file.endswith("jpg")]
        
        for img in imgs:
            dataset.append((np.asarray(Image.open(img)), class_folder))
            #break
        
    flat_dataset = []

    for img_data in dataset:
        flat_dataset.append(img_data[0].ravel())

    X = np.reshape(flat_dataset,(len(flat_dataset),-1))
    # flat_dataset = (flat_dataset - np.mean(flat_dataset))/ np.std(flat_dataset)
    X = scale(X) 
    X_t = PCA(2).fit_transform(X)

    #colors = np.random.rand(len(classes))

    colors = {}

    for label in classes:
        colors[label] = random.random()

    colormap = []

    for i in range(len(dataset)):
        c = colors[dataset[i][1]]
        colormap.append(c)

    plt.grid()
    plt.title("PCA")
    plt.scatter(X_t[:,0],X_t[:,1], c = colormap)
    plt.legend()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()
