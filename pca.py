from os import listdir
from os.path import join
import gc
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


DATA_FOLDER = "./PACS_homework/"


def fetchClasses():
    classes = [ folder for folder in listdir(DATA_FOLDER)]
    print(classes)
    return classes

def project_img(n,X,scaler,rand_int,img_sample_3D,img_shape):
    title = "PCA with the first {} components".format(n)
    pca = PCA(n_components=n)
    X_t = pca.fit_transform(X) # pca_X dimension is 1087 x n 
    
    fig = plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    columns = 2
    rows = 2

    # show standardized reduced image
    X_proj = pca.inverse_transform(X_t)

    temp1 = np.reshape(X_proj[rand_int], img_shape)
    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    plt.title("STD version",y=-0.15)
    plt.imshow(temp1)

    # show non standard version of reduced image
    temp = scaler.inverse_transform(X_proj)
    temp2 = np.reshape(temp[rand_int].astype(int), img_shape)
    fig.add_subplot(rows, columns, 3)
    plt.axis('off')
    plt.title("Non STD version",y=-0.15)
    plt.imshow(temp2)

    fig.add_subplot(rows,columns,4)
    plt.imshow(img_sample_3D)
    plt.axis('off')
    plt.title("original",y=-0.15)
    plt.tight_layout()
    plt.show()


def main():
    classes = fetchClasses()
    classNumber = 1

	# it contains raw pixel of each image
    dataset = []

    for class_folder in classes:
        imgs = [join(DATA_FOLDER,join(class_folder,file)) 
                for file in listdir(join(DATA_FOLDER,class_folder)) 
                if file.endswith("jpg")]
        
        for img in imgs:
			# Read raw pixels of all images
            dataset.append((np.asarray(Image.open(img)), classNumber))
        classNumber += 1
     
    # Convert every image into a 154587-dimensional vector
    vectors_matrix = [img_data[0].ravel() for img_data in dataset]
	# it holds flatten representatio of 3D representation of each image
    # 227 * 227 * 3 -> flatten is 154587
	# it is a N x 154587 matrix(n is the # of images) 
	# rows are samples, columns features

    # Cut 2nd column of dataset in order to get ordinal labels
    columns = list(zip(*dataset))
    labels = np.asarray(columns[1],dtype = np.int16)
    # slice selection: [all] -> [all , column 1]

    

	# from array list -> to ndarray
    #X = np.reshape(vectors_matrix,(len(vectors_matrix),-1))
    # 2nd argument is the shape required and you provide dimensions
    # we want N * 154587. -1 means "evaluate automatically"
    
    rand_int = random.randint(0,len(dataset)-1)
    # 3D image sample
    img_sample_3D = dataset[rand_int][0]
    img_shape = np.asarray(img_sample_3D).shape[:]
    # flatten image sample
    #img_sample_flat = X[rand_int]

    # try to release memory for useless big elements
    #del(vectors_matrix)
    del(dataset)
    gc.collect()


###################### Principal Component Visualization ###############################

    # standardize matrix (mean 0 variance 1)
    scaler = StandardScaler()
    scaler.fit(vectors_matrix) # Compute the mean and std to be used for later scaling.
    X = scaler.transform(vectors_matrix) # standardization = centering and scaling
	# it means subtracts the mean of X from each element (the new mean is 0), 
	# then normalizes the result by the standard deviation
	# The above is for standardizing the entire matrix as a whole.
	# You can standardize each column individually specifying the axis:
	# X = (X - mean(X, axis=0)) / std(X, axis=0)

    project_img(60,X,scaler,rand_int,img_sample_3D,img_shape)
    project_img(6,X,scaler,rand_int,img_sample_3D,img_shape)
    project_img(2,X,scaler,rand_int,img_sample_3D,img_shape)


    pca = PCA().fit(X) # FULL PCA: pca.components is 1087 x 154587 -> it is V transposed
    P = pca.transform(X)
    #Y = Z @ pca.components_[:] == X

    #each row holds an eigenvector
    V_T = pca.components_[-6:,:] # pca_l is 6 x 154587
    #X = X V V^T is the projection matrix
    #give X back his mean and variance in order to fully reconstruct X
    #(in the full original subspace but with last 6 PC)
    Z = X @ V_T.T
    X_copy = Z @ V_T
    X_copy = scaler.inverse_transform(X_copy)
    
    
    fig = plt.figure(figsize=(8, 8))
    temp = np.reshape(X_copy[rand_int].astype(int), img_shape)
    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.title("last 6 PC",y=-0.15)
    plt.tight_layout()
    plt.imshow(temp)

    fig.add_subplot(1,2,2)
    plt.imshow(img_sample_3D)
    plt.axis('off')
    plt.title("original",y=-0.15)
    plt.tight_layout()
    plt.show()
    
    # Scatterplot

    colors = ['red','green','blue','purple']
    row,col = 1,3
    
    # PC 1 AND 2
    plt.subplot(row, col, 1).set_title("PCA first 2 components")
    plt.grid()
    plt.scatter(P[:,0],P[:,1], 
                c=labels, 
                cmap=matplotlib.colors.ListedColormap(colors),
                label = labels,
                alpha=0.73 )
    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,classes,loc=4)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.ylim(-500,500)
    plt.xlim(-600,600)

    # PC 3 AND 4
    plt.subplot(row, col, 2).set_title("PCA 3rd and 4th Component")
    plt.grid()
    plt.scatter(P[:,2],P[:,3], 
                c=labels, 
                cmap=matplotlib.colors.ListedColormap(colors),
                label = labels,
                alpha=0.73 )
    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,classes,loc=4)
    plt.xlabel("Principal Component 3")
    plt.ylabel("Principal Component 4")
    plt.ylim(-500,500)
    plt.xlim(-600,600)

    # PC 10 AND 11
    plt.subplot(row, col, 3).set_title("PCA 10th and 11th Component")
    plt.grid()
    plt.scatter(P[:,9],P[:,10], 
                c=labels, 
                cmap=matplotlib.colors.ListedColormap(colors),
                label = labels,
                alpha=0.73 )
    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,classes,loc=4)
    plt.xlabel("Principal Component 10")
    plt.ylabel("Principal Component 11")
    plt.ylim(-500,500)
    plt.xlim(-600,600)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.close()


    plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
    plt.xlabel('Number of Components') 
    plt.ylabel('Variance retained') 
    plt.ylim(0,1)
    plt.xlim(1,64)
    plt.grid()
    plt.show()



###################################### CLASSIFICATION #########################################
    
    ###########################################################################################
    #
    # No PCA
    #
    X_train, X_test, y_train, y_test = train_test_split(
    X[:,:2], labels, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = (clf.score(X_test,y_test))
    title = "No PCA : Accuracy of {0:.2f} %".format(score*100)

    # Plotting decision regions
    plt.subplot(1, 3, 1).set_title(title)

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = clf.predict(X_test)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=20,label = labels, edgecolor='k')
   
    ###########################################################################################
    #
    # 1st and 2nd component
    #
    pca = PCA(2)
    F_t = pca.fit_transform(X)
    F_proj = pca.inverse_transform(F_t)

    X_train, X_test, y_train, y_test = train_test_split(
    F_proj[:,:2], labels, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = (clf.score(X_test,y_test))
    title = "1st and 2nd PC : Accuracy of {0:.2f} %".format(score*100)

    # Plotting decision regions
    plt.subplot(1, 3, 2).set_title(title)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = clf.predict(X_test)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], label = labels,c=colors, s=20, edgecolor='k')
   

    ###########################################################################################
    #
    # 3rd and 4th component
    #
    pca = PCA().fit(X) 
    V_T = pca.components_[3:5,:] 
    Z = X @ V_T.T
    X_copy = Z @ V_T

    X_train, X_test, y_train, y_test = train_test_split(
    X_copy[:,:2], labels, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = (clf.score(X_test,y_test))
    title = "3rd and 4th PC : Accuracy of {0:.2f} %".format(score*100)

    # Plotting decision regions
    plt.subplot(1, 3, 3).set_title(title)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = clf.predict(X_test)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], label = labels,c=colors, s=20, edgecolor='k')
    # Draw final plot
    plt.suptitle("Naive Bayes Classification")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    return 0

if __name__ == "__main__":
    main()


