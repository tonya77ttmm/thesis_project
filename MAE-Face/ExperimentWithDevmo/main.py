# =========================
# MAIN
# =========================
from .training.grid_search import train_mlp_grid_search
from pathlib import Path
from .datasets.daisee_parser import DaiseeParser
from .datasets.devmo_parser import DevmoParser
from .sampling.sampler import Sampler
from .datasets.frame_dataset import FrameDataset
from .features.mae_extractor import MAEExtractor
from .features.feature_repository import FeatureRepository
from .datasets.feature_dataset import FeatureDataset
from .testing.tester import Tester
import pandas as pd
import os


PROJECT_ROOT=Path(__file__).resolve().parents[2]

def build_dataset(dataset_path, label_path, sampler,parser_class):
    label_df=pd.read_csv(label_path)
    label_dict=dict(zip(label_df['clipID'],label_df['label']))
    parser=parser_class(dataset_path,label_dict,sampler)
    samples=parser.parse()
    dataset=FrameDataset(samples)
    return dataset
    

def extract_features(extractor, batch_size,dataset,feature_dir,save_prefix):
    feature_repository=FeatureRepository(batch_size=batch_size)
    feature_repository.store(dataset,feature_dir,save_prefix, extractor)

    print(f"Features extracted and stored in {feature_dir} with prefix {save_prefix}.")

def build_feature_dataset(feature_dir, save_prefix):
    feature_path = feature_dir / f"{save_prefix}_feats.npy"
    label_path = feature_dir / f"{save_prefix}_labels.npy"
    num_samples = os.path.getsize(feature_path) // (768 * 4)
    return FeatureDataset(feature_path, label_path, num_samples)

def train_mlp(feature_dir, save_prefix, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, num_classes, input_size, num_epochs, device):
    feature_dataset = build_feature_dataset(feature_dir, save_prefix)
    train_mlp_grid_search(input_size, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid,
                                   num_classes=num_classes, num_epochs=num_epochs, device=device, train_dataset=feature_dataset)

    

def evaluate_models():
    pass

if __name__ == "__main__":
    batch_size=64
    device='cuda'
    # img_size = 224
#     load the training dataset and labels
    devmo_dataset_path=PROJECT_ROOT/"confusion_dataset"/"Devmo"/"devemo+"
    devmo_train_label_path=PROJECT_ROOT/"confusion_dataset"/"Devmo"/"devemo+"/"train.csv"
    devmo_test_label_path=PROJECT_ROOT/"confusion_dataset"/"Devmo"/"devemo+"/"test.csv"

    devmo_train_dataset=build_dataset(devmo_dataset_path,devmo_train_label_path,Sampler(fps=15),DevmoParser)
    devmo_test_dataset=build_dataset(devmo_dataset_path,devmo_test_label_path,Sampler(fps=15),DevmoParser)

    mae_extractor=MAEExtractor(ckpt_path=PROJECT_ROOT/"MAE-Face"/"models"/"MAE"/"mae_face_pretrain_vit_base.pth",device=device,feature_dim=768)
    feature_dir=PROJECT_ROOT/"MAE-Face"/"ExperimentWithDevmo"/"data"/"features"
    extract_features(mae_extractor,batch_size,devmo_train_dataset,feature_dir,"devmo_train")
    extract_features(mae_extractor,batch_size,devmo_test_dataset,feature_dir,"devmo_test")

    
    


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
#     devmo_train_feature = feature_dir/"devmo_train_feats.npy"
#     devmo_train_label = feature_dir/"devmo_train_labels.npy"
#     devmo_test_feature = feature_dir/"devmo_test_feats.npy"
#     devmo_test_label = feature_dir/"devmo_test_labels.npy"

    #CV train model on devmo train dataset
    train_mlp(feature_dir, "devmo_train", hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, num_classes, input_size, num_epochs=60, device=device)
    #CV test model on devmo testdataset

    
    
    #CV train model on daisee train dataset
    model_dir=PROJECT_ROOT/"MAE-Face"/"ExperimentWithDevmo"/"data"/"models"
    tester=Tester(build_feature_dataset(feature_dir, "devmo_test"), device)
    for h in hidden_grid:
        print(f"Testing model with hidden layers: {h}")
        tester.test_model(model_dir/f"MLP_best_structure_model_{h}.pth",h)

    #CV test model on daisee test dataset

    #CV model on devmo train dataset and test on daisee test dataset

    #CV model on daisee train dataset and test on devmo test dataset

    #train model on mixed dataset and test seperately on devmo and daisee test datasets



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

    

    
