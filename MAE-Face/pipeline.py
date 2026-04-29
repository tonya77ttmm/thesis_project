import subprocess


if __name__=="__main__":


    # Run feature extraction
    subprocess.run(["python", "feature_Extraction.py"], check=True)

    # Run MLP model
    subprocess.run(["python", "mlp.py"], check=True)
    print("Pipeline completed. You can now evaluate the trained MLP models using the saved checkpoints.")