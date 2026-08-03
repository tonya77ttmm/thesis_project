# =========================
# MAIN
# =========================
from .experiments.grid_search import train_mlp_grid_search
from pathlib import Path
from .datasets.daisee_parser import DaiseeParser
from .datasets.devmo_parser import DevmoParser
from .sampling.sampler import Sampler
from .datasets.frame_dataset import FrameDataset
from .features.mae_extractor import MAEExtractor
from .features.feature_repository import FeatureRepository
from .datasets.feature_dataset import FeatureDataset
import pandas as pd
import os

PROJECT_ROOT=Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    batch_size=64
    #settings for video, model load path

    device='cuda'
    # img_size = 224
    #load the training dataset and labels
    # devmo_dataset_path=PROJECT_ROOT/"confusion_dataset"/"Devmo"/"devemo+"
    # devmo_train_df=pd.read_csv(devmo_dataset_path/"train.csv")
    # devmo_train_label_dict=dict(zip(devmo_train_df['clipID'], devmo_train_df['label']))
    # devmo_test_df=pd.read_csv(devmo_dataset_path/"test.csv")
    # devmo_test_label_dict=dict(zip(devmo_test_df['clipID'], devmo_test_df['label']))    

    # sampler=Sampler(fps=15)
    # devmo_train_samples_parser=DevmoParser(devmo_dataset_path,devmo_train_label_dict,sampler)
    # devmo_train_samples=devmo_train_samples_parser.parse()
    # #img, label type(tensor)
    # devmo_train_frame_dataset=FrameDataset(devmo_train_samples)
    # mae_extractor=MAEExtractor(ckpt_path=PROJECT_ROOT/"MAE-Face"/"models"/"MAE"/"mae_face_pretrain_vit_base.pth",device=device,feature_dim=768)
    # feature_repository=FeatureRepository(batch_size=batch_size)
    feature_dir=PROJECT_ROOT/"MAE-Face"/"ExperimentWithDevmo"/"data"/"features"
    # feature_repository.store(devmo_train_frame_dataset,feature_dir,"devmo_train", mae_extractor)

    
    print("Feature extraction completed. You can now train the MLP models using the extracted features saved in the ./Features directory.")


    # hidden_grid = [
    #     [64], [128], [128, 64],
    #     [256], [256, 128], [256, 128, 64],
    #     [512], [1024], [512, 256], [1024, 512]
    # ]
    hidden_grid = [
            [64], 
            [256], [256, 128], 
            [512], [1024]
    ]
     
    lr_grid = [1e-3, 1e-4]
    wd_grid = [1e-2,1e-3, 1e-4]
    thresh_grid = [0.3,0.4, 0.5, 0.6,0.7, 0.8]
    drop_grid=[0.2,0.3, 0.4,0.5]
    num_classes=2
    input_size = 768
    # =========================
    # devmo feature dataset
    # =========================
    devmo_train_feature = feature_dir/"devmo_train_feats.npy"
    devmo_train_label = feature_dir/"devmo_train_labels.npy"
    num_devmo_train_samples = os.path.getsize(devmo_train_feature) // (768 * 4)

    devmo_train_feature_dataset=FeatureDataset(devmo_train_feature,devmo_train_label,num_devmo_train_samples)

    train_mlp_grid_search(input_size, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, 
                                   num_classes=2, num_epochs=60, device='cuda', train_dataset=devmo_train_feature_dataset)
    # A_test_feat  = f"{data_dir}/Devmo-test_feats.npy"
    # A_test_label = f"{data_dir}/Devmo-test_labels.npy"

    # =========================
    # daisee dataset
    # # =========================
    # B_train_feat = f"{data_dir}/Train_v2_feats_cc.npy"
    # B_train_label = f"{data_dir}/Train_v2_labels_cc.npy"
    # B_test_feat  = f"{data_dir}/Val_feats_cc.npy"
    # B_test_label = f"{data_dir}/Val_labels_cc.npy"

    # counts
    # num_A_train = os.path.getsize(A_train_feat) // (768 * 4)
    # num_A_test  = os.path.getsize(A_test_feat) // (768 * 4)
    # num_B_train = os.path.getsize(B_train_feat) // (768 * 4)
    
    # num_B_test  = os.path.getsize(B_test_feat) // (768 * 4)

    # train_dataset = ConcatDataset([
    #         FeatureDataset(A_train_feat, A_train_label, num_A_train),
    #         FeatureDataset(B_train_feat, B_train_label, num_B_train)
    #     ])

    #grid search for mixed_ train_dataset
    

    # train_mlp_grid_search(
    #     input_size=input_size,
    #     hidden_grid=hidden_layer_variants,
    #     lr_grid=learning_rates,
    #     wd_grid=weight_decays,
    #     drop_grid=dropout_rates,
    #     thresh_grid=thresholds,
    #     num_classes=num_classes,
    #     num_epochs=60,
    #     device=device,
    #     train_dataset=train_dataset
    # )

    
    # #最后做
    # devmo_test_dataset=FeatureDataset(A_test_feat, A_test_label, num_A_test)
    # daisee_test_dataset=FeatureDataset(B_test_feat, B_test_label, num_B_test)

   
    # final_training_and_evaluation(
    #     hidden_grid=hidden_layer_variants, 
    #     train_dataset=train_dataset,
    #     devmo_test_dataset=devmo_test_dataset,
    #     daisee_test_dataset=daisee_test_dataset,
    #     device=device
    #     )


    #just test for all features in val set in daisee
    
    # B_test_feat_all  = f"{data_dir}/Val-all_feats_cc.npy"
    # B_test_label_all = f"{data_dir}/Val-all_labels_cc.npy"
    # num_B_test_all  = os.path.getsize(B_test_feat_all) // (768 * 4)
    # daisee_test_dataset_all=FeatureDataset(B_test_feat_all, B_test_label_all, num_B_test_all)
    # evaluate_final_model(hidden_layer_variants, daisee_test_dataset, device)

    #==================
    #daisee dataset 
    #==================
    # train_dataset_path="../confusion_dataset/DAiSEE/DataSet/Train/"
    # train_Labels_v1="../confusion_dataset/DAiSEE/Labels/TrainLabels_confusion.csv"
    # train_df_v1=pd.read_csv(train_Labels_v1)
    # train_label_dict_v1=dict(zip(train_df_v1['ClipID'], train_df_v1['Confusion']))

    # train_labels_v2="../confusion_dataset/DAiSEE/Labels/4_TrainLabels_confusion.csv"
    # train_df_v2=pd.read_csv(train_labels_v2)
    # train_label_dict_v2=dict(zip(train_df_v2['ClipID'], train_df_v2['Confusion']))

    
    # #load the validation dataset and labels
    # val_dataset_path="../confusion_dataset/DAiSEE/DataSet/Validation/"
    # val_Labels="../confusion_dataset/DAiSEE/Labels/ValidationLabels_confusion.csv"
    # val_df=pd.read_csv(val_Labels)
    # val_label_dict=dict(zip(val_df['ClipID'], val_df['Confusion']))

    

    
