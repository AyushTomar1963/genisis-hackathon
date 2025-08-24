# Cell 9: Time Series Validation and Data Leakage Check

# Alternative: Time Series Cross-Validation (if you want more robust validation)
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(X, y_class, y_reg, n_splits=5):
    """
    Perform time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores_class = []
    cv_scores_reg = []
    
    fold = 1
    for train_idx, val_idx in tscv.split(X):
        print(f"Fold {fold}: Train size={len(train_idx)}, Val size={len(val_idx)}")
        
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_class_train_cv, y_class_val_cv = y_class[train_idx], y_class[val_idx]
        y_reg_train_cv, y_reg_val_cv = y_reg[train_idx], y_reg[val_idx]
        
        # Create a smaller model for CV (to save time)
        cv_model = build_multitask_lstm(X_train_cv.shape[1:], lstm_units=64, dropout_rate=0.2)
        cv_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
            loss_weights={'classification': 1.0, 'regression': 0.5},
            metrics={'classification': ['accuracy'], 'regression': ['mae']}
        )
        
        # Train with fewer epochs for CV
        cv_model.fit(
            X_train_cv, 
            {'classification': y_class_train_cv, 'regression': y_reg_train_cv},
            epochs=20, 
            batch_size=32, 
            verbose=0
        )
        
        # Evaluate
        cv_pred = cv_model.predict(X_val_cv, verbose=0)
        cv_class_acc = accuracy_score(y_class_val_cv, (cv_pred[0] > 0.5).astype(int))
        cv_reg_mse = mean_squared_error(y_reg_val_cv, cv_pred[1])
        
        cv_scores_class.append(cv_class_acc)
        cv_scores_reg.append(cv_reg_mse)
        
        print(f"  Classification Accuracy: {cv_class_acc:.4f}")
        print(f"  Regression MSE: {cv_reg_mse:.4f}")
        fold += 1
    
    return cv_scores_class, cv_scores_reg

# Option to run cross-validation (comment out if you want to skip)
print("TIME SERIES CROSS-VALIDATION:")
print("="*50)
print("Note: This will take several minutes. Set run_cv=True to execute.")
run_cv = False  # Set to True if you want to run CV

if run_cv:
    cv_class_scores, cv_reg_scores = time_series_cross_validation(X, y_class, y_reg)
    
    print(f"\nCross-Validation Results:")
    print(f"Classification Accuracy: {np.mean(cv_class_scores):.4f} ± {np.std(cv_class_scores):.4f}")
    print(f"Regression MSE: {np.mean(cv_reg_scores):.4f} ± {np.std(cv_reg_scores):.4f}")
else:
    print("Cross-validation skipped. Set run_cv=True to execute.")

# Data Leakage Check
print(f"\nDATA LEAKAGE CHECK:")
print("="*50)

# Check if any training sequences overlap with test sequences
train_end_time = split_index + sequence_length - 1
test_start_time = split_index

print(f"Last training sequence ends at index: {train_end_time}")
print(f"First test sequence starts at index: {test_start_time}")
print(f"Temporal gap: {test_start_time - train_end_time} time steps")

if train_end_time >= test_start_time:
    print("⚠️ WARNING: Potential data leakage detected!")
    print("   Training sequences overlap with test period.")
    print("   Consider increasing the gap or adjusting sequence length.")
else:
    print("✅ No data leakage detected. Clean temporal split.")

# Visualize the temporal split
plt.figure(figsize=(15, 8))

# Create a timeline showing train/test split
total_samples = len(df)
timeline = np.arange(total_samples)

plt.subplot(2, 1, 1)
plt.plot(timeline[:split_index], np.ones(split_index), 'b-', linewidth=3, label='Training Data')
plt.plot(timeline[split_index:], np.ones(total_samples - split_index), 'r-', linewidth=3, label='Test Data')
plt.axvline(x=split_index, color='green', linestyle='--', linewidth=2, label='Split Point')
plt.title('Temporal Train-Test Split')
plt.xlabel('Time Index')
plt.ylabel('Dataset')
plt.legend()
plt.grid(True, alpha=0.3)

# Show actual vs predicted on test set for a sample
sample_size = min(200, len(y_reg_test))
plt.subplot(2, 1, 2)
plt.plot(y_reg_test_original[:sample_size], 'b-', label='Actual', alpha=0.7)
plt.plot(y_reg_pred_original[:sample_size], 'r-', label='Predicted', alpha=0.7)
plt.fill_between(range(sample_size), 
                 y_reg_pred_original[:sample_size] - np.std(residuals[:sample_size]),
                 y_reg_pred_original[:sample_size] + np.std(residuals[:sample_size]),
                 alpha=0.2, color='red', label='Prediction Interval')
plt.title(f'Time Series Prediction on Test Set (First {sample_size} samples)')
plt.xlabel('Time Index')
plt.ylabel('Traffic Volume')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional validation: Walk-forward validation snippet
def walk_forward_validation(X, y_class, y_reg, n_steps=5):
    """
    Simple walk-forward validation for time series
    """
    total_samples = len(X)
    step_size = (total_samples - split_index) // n_steps
    
    scores_class = []
    scores_reg = []
    
    for i in range(n_steps):
        # Expanding window: always train from beginning
        train_end = split_index + i * step_size
        test_start = train_end
        test_end = min(test_start + step_size, total_samples)
        
        if test_end - test_start < 10:  # Skip if test set too small
            continue
            
        X_train_wf = X[:train_end]
        X_test_wf = X[test_start:test_end]
        y_class_train_wf = y_class[:train_end]
        y_class_test_wf = y_class[test_start:test_end]
        y_reg_train_wf = y_reg[:train_end]
        y_reg_test_wf = y_reg[test_start:test_end]
        
        # Use pre-trained model for quick validation
        wf_pred = model.predict(X_test_wf, verbose=0)
        wf_class_acc = accuracy_score(y_class_test_wf, (wf_pred[0] > 0.5).astype(int))
        wf_reg_mse = mean_squared_error(y_reg_test_wf, wf_pred[1])
        
        scores_class.append(wf_class_acc)
        scores_reg.append(wf_reg_mse)
        
        print(f"Step {i+1}: Train={train_end}, Test={test_start}-{test_end}, "
              f"Acc={wf_class_acc:.4f}, MSE={wf_reg_mse:.4f}")
    
    return scores_class, scores_reg

print(f"\nWALK-FORWARD VALIDATION:")
print("-"*50)
wf_class_scores, wf_reg_scores = walk_forward_validation(X, y_class, y_reg)

if wf_class_scores:
    print(f"\nWalk-Forward Results:")
    print(f"Classification Accuracy: {np.mean(wf_class_scores):.4f} ± {np.std(wf_class_scores):.4f}")
    print(f"Regression MSE: {np.mean(wf_reg_scores):.4f} ± {np.std(wf_reg_scores):.4f}")
    print(f"Stability: Classification CV = {np.std(wf_class_scores)/np.mean(wf_class_scores)*100:.1f}%")
    print(f"           Regression CV = {np.std(wf_reg_scores)/np.mean(wf_reg_scores)*100:.1f}%")