def plot_residuals():
    import matplotlib.pyplot as plt

    # Assuming you have already trained your model and made predictions
    # y_train_pred and y_test_pred contain the predicted values

    # Create a single figure with both scatter plots and residual plots
    plt.figure(figsize=(12, 6))

    # Training set scatter plot with red diagonal line
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', lw=2)
    plt.title("Training Set: Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    # Testing set scatter plot with red diagonal line
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
    plt.title("Testing Set: Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    # Add a new row for residual plots
    plt.figure(figsize=(12, 6))

    # Training set residual plot
    plt.subplot(2, 2, 3)
    plt.scatter(y_train_pred, y_train - y_train_pred)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title("Training Set: Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    # Testing set residual plot
    plt.subplot(2, 2, 4)
    plt.scatter(y_test_pred, y_test - y_test_pred)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title("Testing Set: Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    plt.tight_layout()
    plt.show()