from pathlib import Path
from ..training.cross_validator import CrossValidator
from ..training.evaluator import Evaluator
import torch
def train_mlp_grid_search(input_size, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, 
                           num_classes=2, num_epochs=60, device='cuda', train_dataset=None):
    PROJECT_ROOT=Path(__file__).resolve().parents[1]
    
    csv_results_records = []
    evaluator=Evaluator(device)
    cross_validator=CrossValidator(train_dataset,device,thresh_grid,num_epochs,evaluator)
    # ======================================================================
    # LAYER 1: Loop over different Network Hidden Layer Structures (e.g., [32] vs [64, 32])
    # ======================================================================
    for h in hidden_grid:
        hidden_str = "_".join(map(str, h))
        
        # Tracks the absolute best hyperparameter combination for this specific structure
        structure_best_f1 = -1
        structure_best_model_state = None
        structure_best_meta = {}

        # ======================================================================
        # LAYERS 2-4: Loop over training-time hyperparameters (The "Combo")
        # All 5 folds will be processed under this fixed combination.
        # ======================================================================
        for lr in lr_grid:
            for wd in wd_grid:
                for drop in drop_grid:
                    print(f" Running Config -> LR: {lr} | WD: {wd} | Dropout: {drop}")

                    combo_results, combo_best_fold0_model=cross_validator.evaluate_combo(
                        architecture=h, lr=lr, wd=wd, drop=drop, input_size=input_size,  num_classes=num_classes, 
                    )
                    csv_results_records.append(combo_results)

                    # Check if this entire hyperparameter combination beats previous combinations tried under this architecture
                    if combo_results["combo_best_metrics"].f1 > structure_best_f1:
                        structure_best_f1 = combo_results["combo_best_metrics"].f1
                        structure_best_meta = {
                            'lr': lr, 'wd': wd, 'dropout': drop, 'best_threshold': combo_results['best_threshold'],
                            'structure_results':combo_results["combo_best_metrics"]
                        }
                        # Save these specific weights as the absolute best version of this architecture structure
                        structure_best_model_state = combo_best_fold0_model

        # Save exactly one `.pth` model file for the current structural layer architecture
        if structure_best_model_state is not None:
            save_path = PROJECT_ROOT/"data"/"models"/f"MLP_best_structure_model_{hidden_str}.pth"
            torch.save({
                'model_state_dict': structure_best_model_state,
                'architecture': h,
                'hyperparameters': structure_best_meta
            }, save_path)
            print(f"\n [SAVED MODEL] Top performing model file saved for structure {h} -> {save_path}")

    # Export unique combination statistics to the final CSV file
    results_df = pd.DataFrame(csv_results_records)
    csv_file_path = PROJECT_ROOT/"data"/"results"/f"mlp_grid_search_results.csv"
    results_df.to_csv(csv_file_path, index=False)
    print(f"\n[SAVED CSV LOG] Grid search results file saved to: {csv_file_path}\n")
                   




