def evaluate_final_model(hidden_layer_variants, test_dataset, device):
    
    for h in hidden_layer_variants:
        print("\n====================")
        print("Structure:", h)

        ckpt_path=f"models/MLP/mixed/MLP_{'_'.join(map(str,h))}_final_model.pth"
        ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp=ckpt["hyperparameters"]
        print(hp)
        model=EmotionMLP(input_size=768, hidden_layers=h, dropout_rate=hp["dropout"], num_classes=2).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loader_B_all=DataLoader(test_dataset, batch_size=64)
       
        metrics_B=test_model(model, test_loader_B_all, hp["best_threshold"], device)

        print("B:", metrics_B)