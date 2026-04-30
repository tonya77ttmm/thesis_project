import os
import subprocess
import cv2
import shutil
import pandas as pd
OPEN_FACE_DIR="../OpenFace/build/bin/FaceLandmarkImg"

#this is slow
def upscale_folder(extract_path, upscale_path, scale_factor=2):
    os.makedirs(upscale_path,exist_ok=True)
    for frame in os.listdir(extract_path):
        if frame.endswith(".png"):
            img=cv2.imread(os.path.join(extract_path,frame))
            img_resize=cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imwrite(os.path.join(upscale_path,frame), img_resize)

def flatten_openface_output(faces_path):
    #iterate all the csv file in the faces path
    for file in os.listdir(faces_path):
        if file.endswith(".csv"):
            file_number=file.replace(".csv","")
            csv_file_path=os.path.join(faces_path,file)
            df=pd.read_csv(csv_file_path)

            #Find face with highest confidence
            best_face_idx=df['confidence'].idxmax()
            best_face_number=int(df.loc[best_face_idx,'face'])

            #find the corresponding bmp file in the corresponding aligned folder
            aligned_folder=os.path.join(faces_path,file_number+"_aligned")
            best_face_file=os.path.join(aligned_folder,f"face_det_{best_face_number:06d}.bmp")

            #move the bmp file to the faces path 
            if os.path.isfile(best_face_file):
                new_face_file=os.path.join(faces_path,f"face_{file_number}.bmp")
                shutil.move(best_face_file,new_face_file)

            #delete the aligned folder
            shutil.rmtree(aligned_folder)
           

    # 

    # counter=0
    # for root,dirs,files in os.walk(faces_path):
    #     for file in files:
    #         if file.endswith(".bmp"):
    #             src=os.path.join(root,file)
    #             dst=os.path.join(faces_path,f"face_{counter}.bmp")
    #             shutil.move(src,dst)
    #             counter+=1
    # for root,dirs,files in os.walk(faces_path):
    #     for d in dirs:
    #         if "aligned" in d:
    #             shutil.rmtree(os.path.join(root,d))


# def extract_face_landmarks_daisee(dataset_dir):
    for usr in os.listdir(dataset_dir):
        if usr in unfinished_usr_training_folders:
            usr_path=os.path.join(dataset_dir,usr)
            for extract in os.listdir(usr_path):
                extract_path=os.path.join(usr_path,extract)
                upscale_path=os.path.join(extract_path,"upscale")
                upscale_folder(extract_path, upscale_path,2)

                faces_path=os.path.join(extract_path,"openFaces")
                os.makedirs(faces_path,exist_ok=True)

                command = [OPEN_FACE_DIR, "-fdir", upscale_path, "-out_dir", faces_path]
                subprocess.run(command)
            
                print(f"Finished processing {extract_path}")
                flatten_openface_output(faces_path)
                
def extract_face_landmarks_daisee(dataset_dir):
    for usr in os.listdir(dataset_dir):
        usr_path=os.path.join(dataset_dir,usr)
        for extract in os.listdir(usr_path):
            extract_path=os.path.join(usr_path,extract)
            upscale_path=os.path.join(extract_path,"upscale")
            upscale_folder(extract_path, upscale_path,2)

            faces_path=os.path.join(extract_path,"openFaces")
            os.makedirs(faces_path,exist_ok=True)

            command = [OPEN_FACE_DIR, "-fdir", upscale_path, "-out_dir", faces_path]
            subprocess.run(command)
            
            print(f"Finished processing {extract_path}")
            flatten_openface_output(faces_path)

                

if __name__ == "__main__":
    training_dataset_dir="../../confusion_dataset/DAiSEE/DataSet/Train/"
    extract_face_landmarks_daisee(training_dataset_dir)
    #extract_face_landmarks(training_dataset_dir)
    #print(f"Finished extracting training dataset")


    validation_dataset_dir="../../confusion_dataset/DAiSEE/DataSet/Validation/"
    extract_face_landmarks_daisee(validation_dataset_dir)
    
    test_dataset_dir="../../confusion_dataset/DAiSEE/DataSet/Test/"
    extract_face_landmarks_daisee(test_dataset_dir)
    
    print(f"Finished extracting validation dataset")

