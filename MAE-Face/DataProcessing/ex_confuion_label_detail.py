import pandas as pd


def extract_confusion_statistics(src,dst):
    df = pd.read_csv(src)
    confusion_counts=df['Confusion'].value_counts()
    for value, count in confusion_counts.items():
        print(f"Confusion value {value}: {count} samples")


def extract_confusion_label(src,dst):
    # Load the CSV
    df = pd.read_csv(src)
    
    # Binarize the Confusion label
    # # 0 or 1 -> 0, 2 or 3 -> 1
    # df['Confusion'] = df['Confusion'].apply(lambda x: 1 if x >= 2 else 0)
    confusion_counts=df['Confusion'].value_counts()
    for value, count in confusion_counts.items():
        print(f"Confusion value {value}: {count} samples")
    # Keep only ClipID and Confusion columns
    df_new = df[['ClipID', 'Confusion']]

    # Save to a new CSV
    df_new.to_csv(dst, index=False)

    print(f"New CSV saved as '{dst}'")
# # Load the CSV
# df = pd.read_csv('../../confusion_dataset/DAiSEE/Labels/TrainLabels.csv')

# # Binarize the Confusion label
# # 0 or 1 -> 0, 2 or 3 -> 1
# df['Confusion'] = df['Confusion'].apply(lambda x: 1 if x >= 2 else 0)

# # Keep only ClipID and Confusion columns
# df_new = df[['ClipID', 'Confusion']]

# # Save to a new CSV
# df_new.to_csv('../../confusion_dataset/DAiSEE/Labels/TrainLabels_confusion.csv', index=False)

# print("New CSV saved as 'trainlabels_confusion.csv'")

if __name__ == "__main__":
    #extact_training_confusion_label
    # extract_confusion_label('../../confusion_dataset/DAiSEE/Labels/TrainLabels.csv','../../confusion_dataset/DAiSEE/Labels/4_TrainLabels_confusion.csv')
    # # extact_validation_confusion_label
    # extract_confusion_label('../../csonfusion_dataset/DAiSEE/Labels/ValidationLabels.csv','../../confusion_dataset/DAiSEE/Labels/4_ValidationLabels_confusion.csv')
    # #extact_test_confusion_label            
    # extract_confusion_label('../../confusion_dataset/DAiSEE/Labels/TestLabels.csv','../../confusion_dataset/DAiSEE/Labels/TestLabels_confusion.csv')

    
    extract_confusion_statistics('../../confusion_dataset/DAiSEE/Labels/TrainLabels.csv','../../confusion_dataset/DAiSEE/Labels/4_TrainLabels_confusion.csv')

    extract_confusion_statistics('../../confusion_dataset/DAiSEE/Labels/ValidationLabels.csv','../../confusion_dataset/DAiSEE/Labels/4_ValidationLabels_confusion.csv')

    extract_confusion_statistics('../../confusion_dataset/DAiSEE/Labels/TestLabels.csv','../../confusion_dataset/DAiSEE/Labels/TestLabels_confusion.csv')