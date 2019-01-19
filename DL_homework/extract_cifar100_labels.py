def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_names = unpickle("./data/cifar-100-python/meta")

f = open("label_coarse.txt", "w")
for label in label_names[b'coarse_label_names']:
    f.write(label.decode("utf-8") + "\n")
f.close()

f = open("label_fine.txt", "w")
for label in label_names[b'fine_label_names']:
    f.write(label.decode("utf-8") + "\n")

