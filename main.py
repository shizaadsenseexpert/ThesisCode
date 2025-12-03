"""
Main Execution Script for SE-CNN Model
Complete pipeline for SE-CNN-based Marketing Optimization
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src import (
    set_seeds, create_project_structure, save_experiment_config,
    preprocess_and_split, create_model, train_model, evaluate_model,
    optimize_budget
)
from src.utils import print_banner, generate_summary_report


def main():
    """Main execution function"""
    
    # Print project header
    print_banner("SE-CNN-BASED MARKETING OPTIMIZATION FRAMEWORK")
    print("Model: Squeeze-and-Excitation CNN with Channel Attention\n")
    
    # Step 1: Setup
    print_banner("STEP 1: PROJECT SETUP")
    set_seeds()
    create_project_structure()
    save_experiment_config()
    
    # Step 2: Data Preprocessing
    print_banner("STEP 2: DATA PREPROCESSING")
    data = preprocess_and_split()
    
    X_train_num, X_train_cat, y_train_class, y_train_reg, train_df = data['train']
    X_val_num, X_val_cat, y_val_class, y_val_reg, val_df = data['val']
    X_test_num, X_test_cat, y_test_class, y_test_reg, test_df = data['test']
    preprocessor = data['preprocessor']
    
    print(f"\nData split completed:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Testing: {len(test_df)} samples")
    
    # Step 3: Model Building
    print_banner("STEP 3: SE-CNN MODEL ARCHITECTURE")
    print("Building model with Squeeze-and-Excitation blocks...")
    model_builder = create_model(preprocessor.feature_dims)
    model = model_builder.model
    
    # Step 4: Model Training
    print_banner("STEP 4: MODEL TRAINING")
    train_data = (X_train_num, X_train_cat, y_train_class, y_train_reg)
    val_data = (X_val_num, X_val_cat, y_val_class, y_val_reg)
    
    trainer, train_metrics = train_model(model, train_data, val_data)
    
    # Step 5: Model Evaluation
    print_banner("STEP 5: MODEL EVALUATION")
    test_data = (X_test_num, X_test_cat, y_test_class, y_test_reg, test_df)
    evaluator, comparison_df = evaluate_model(model, test_data)
    
    # Step 6: Budget Optimization
    print_banner("STEP 6: BUDGET OPTIMIZATION (INITIALIZING)")
    optimizer = None
    opt_results = None
    recommendations = None
    allocation_message = "NOT RUN"
    try:
        result = optimize_budget(evaluator.predictions, test_df, preprocessor.label_encoders)
        if result is not None:
            optimizer, recommendations = result
            opt_results = optimizer.optimization_results
            strategy = getattr(optimizer, 'allocation_strategy', None)
            if strategy == 'constraint':
                allocation_message = "CONSTRAINT ALLOCATION"
            elif strategy == 'flexible':
                allocation_message = "FLEXIBLE ALLOCATION"
            else:
                allocation_message = "UNSPECIFIED ALLOCATION"
        else:
            print("Budget optimization skipped due to strategy failures")
            allocation_message = "NO VALID ALLOCATION"
    except Exception as e:
        print(f"Budget optimization failed: {str(e)}")
        allocation_message = "FAILED"
    finally:
        print_banner(f"STEP 6: BUDGET OPTIMIZATION ({allocation_message})")
    
    # Step 7: Generate Summary Report
    print_banner("STEP 7: GENERATING SUMMARY REPORT")
    generate_summary_report(
        train_metrics=train_metrics,
        eval_metrics=evaluator.metrics,
        opt_results=opt_results
    )
    
    # Final message
    print_banner("PIPELINE COMPLETED SUCCESSFULLY!")
    print("\nAll results saved to 'se_cnn_model/results/' directory:")
    print("  - Model: se_cnn_model/models/se_cnn_marketing_model.h5")
    print("  - Training plots: se_cnn_model/results/training_history.png")
    print("  - Confusion matrix: se_cnn_model/results/confusion_matrix.png")
    print("  - ROC curve: se_cnn_model/results/roc_curve.png")
    print("  - Regression results: se_cnn_model/results/regression_results.png")
    if optimizer is not None:
        print("  - Budget allocation: se_cnn_model/results/budget_allocation.png")
        print("  - Budget recommendations: se_cnn_model/results/budget_recommendations.csv")
    print("  - Model comparison: se_cnn_model/results/model_comparison.csv")
    print("  - Summary report: se_cnn_model/results/summary_report.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print('='*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print('='*80 + "\n")
        sys.exit(1)

