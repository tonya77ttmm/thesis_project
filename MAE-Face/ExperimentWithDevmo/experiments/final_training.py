
# =========================
# train function (IMPORTANT)
# =========================
def train_model(model, train_dataset, lr, wd, epochs, device):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    all_labels=np.concatenate([train_dataset.datasets[0].labels, train_dataset.datasets[1].labels])
    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loader=DataLoader(train_dataset, batch_size=64,shuffle=True)

    model.train()

    for ep in range(epochs):
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1}/{epochs} | loss={total_loss/len(train_loader):.4f}")

def final_training_and_evaluation(hidden_grid, train_dataset,devmo_test_dataset, daisee_test_dataset, device):
    results=[]
    for h in hidden_grid:
        print("\n====================")
        print("Structure:", h)

        ckpt_path=f"models/MLP/mixed/MLP_{'_'.join(map(str,h))}_best_structure_model.pth"
        ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp=ckpt["hyperparameters"]
        model=EmotionMLP(input_size=768, hidden_layers=h, dropout_rate=hp["dropout"], num_classes=2).to(device)
        
        train_model(model, train_dataset, lr=hp["lr"], wd=hp["wd"], epochs=hp['structure_best_epoch'], device=device)
        test_loader_A=DataLoader(devmo_test_dataset, batch_size=64)
        metrics_A=test_model(model, test_loader_A, hp["best_threshold"], device)
        test_loader_B=DataLoader(daisee_test_dataset, batch_size=64)
        metrics_B=test_model(model, test_loader_B, hp["best_threshold"], device)

        save_path=f"models/MLP/mixed/MLP_{'_'.join(map(str,h))}_final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': h,
            'hyperparameters': hp, 
        }, save_path)
        results.append({
            "structure": str(h),
            "metrics_A": metrics_A,
            "metrics_B": metrics_B
        })
        print("A:", metrics_A)
        print("B:", metrics_B)

    #save CSV
    df=pd.DataFrame(results)
    df.to_csv("A_B_joint_training_final_results.csv", index=False)