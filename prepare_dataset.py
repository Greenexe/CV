from PIL import Image
import os
import numpy as np
from sklearn import preprocessing

def main():
    train_filepath = "C:/Users/Jakub/Desktop/Portfolio/GAN/ImageNEt/ImageNet100/train/"
    out_filename = "mageNet100x32.npz"
    img_size = 32

    # Load data
    dataset = []
    filepaths = []
    print("Loading...")
    print("here")
    for folders, sub_folders, files in os.walk(train_filepath):
        # For each name in the files
        
        for filename in files:
            
            dataset.append(os.path.join(folders, filename))
            filepaths.append(folders.split("/")[-1])

    # Convert data
    #Images
    images = []
    iterator = 1
    maximum = len(dataset)
    print("Converting...")
    for path in dataset:
        print("\r",iterator,"/",maximum,end="")
        img = Image.open(path).convert('RGB')
        img = img.resize((img_size, img_size))
        arr = np.array(img, dtype=np.uint8)
        images.append(arr)
        iterator += 1
    images = np.array(images, dtype=np.uint8)

    #Filepaths
    filepaths = np.array(filepaths)
    le = preprocessing.LabelEncoder()
    le.fit(filepaths)
    filepaths = np.reshape(le.transform(filepaths), (-1, 1))
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(filepaths)
    filepaths = enc.transform(filepaths).toarray()
    filepaths = np.array(filepaths, dtype=np.uint8)

    print("")
    print("Images shape:",images.shape,", Filepaths shape:",filepaths.shape)

    # Save arrays
    print("Saving...")
    np.savez(out_filename, images, filepaths)
        


if __name__ == "__main__":
    main()