
#delete all jpg files under each extract
#delete upscale folder under each extractto free up space
#whole extract: 111M.  OpenFaces:69M  Upscale:39M
#it will release half of the space (111-69)/111=38% space released

import os
import shutil


def release_space(dataset_path):
    for usr in os.listdir(dataset_path):
        usr_path=os.path.join(dataset_path,usr)
        for extract in os.listdir(usr_path):
            extract_path=os.path.join(usr_path,extract)
            #delete all jpg files under each extract
            for file in os.listdir(extract_path):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(extract_path,file))
            #delete upscale folder under each extract
            upscale_folder=os.path.join(extract_path,"upscale")
            openFaces_folder=os.path.join(extract_path,"openFaces")
            if os.path.exists(upscale_folder):
                shutil.rmtree(upscale_folder)
            #delete all jpg files under openFaces
            for file in os.listdir(openFaces_folder):
                if file.endswith(".jpg"):
                    os.remove(os.path.join(openFaces_folder,file))
            print(f"Released space for {extract_path}")

if __name__=="__main__":
    # train_dataset_path="../../confusion_dataset/DAiSEE/DataSet/Train/"
    # val_dataset_path="../../confusion_dataset/DAiSEE/DataSet/Validation/"
    test_dataset_path="../../confusion_dataset/DAiSEE/DataSet/Test/"
    # release_space(train_dataset_path)
    # release_space(val_dataset_path)
    release_space(test_dataset_path)

    print("Space released for both training and validation datasets")
   