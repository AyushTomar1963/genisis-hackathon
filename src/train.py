# Cell 5: Train the Model
# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=8, 
    min_lr=0.0001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Train the model
print("Starting model training...")
history = model.fit(
    X_train,
    {
        'classification': y_class_train,
        'regression': y_reg_train
    },
    validation_data=(
        X_test,
        {
            'classification': y_class_test,
            'regression': y_reg_test
        }
    ),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("Training completed!")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Classification accuracy
axes[0, 0].plot(history.history['classification_accuracy'], label='Train Accuracy')
axes[0, 0].plot(history.history['val_classification_accuracy'], label='Val Accuracy')
axes[0, 0].set_title('Classification Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Classification loss
axes[0, 1].plot(history.history['classification_loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_classification_loss'], label='Val Loss')
axes[0, 1].set_title('Classification Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Regression MAE
axes[1, 0].plot(history.history['regression_mae'], label='Train MAE')
axes[1, 0].plot(history.history['val_regression_mae'], label='Val MAE')
axes[1, 0].set_title('Regression MAE')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Total loss
axes[1, 1].plot(history.history['loss'], label='Train Total Loss')
axes[1, 1].plot(history.history['val_loss'], label='Val Total Loss')
axes[1, 1].set_title('Total Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Save the model
model.save('multitask_lstm_network_model.h5')
print("Model saved as 'multitask_lstm_network_model.h5'")