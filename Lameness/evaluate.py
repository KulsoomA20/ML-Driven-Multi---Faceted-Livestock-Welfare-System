import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load dataset of features and labels
features_df = pd.read_csv('features.csv') 

# Separate features and labels
X = features_df.drop(columns=['label', 'video']).values
y = features_df['label'].values

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reshape for LSTM input: (samples, timesteps, features)
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

# Create train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Load trained LSTM model
model = tf.keras.models.load_model('lstm_model_tuned1.h5')  

# Predict class probabilities on test set
pred_probs = model.predict(X_test)

# Convert probabilities to predicted class indices
pred_classes = pred_probs.argmax(axis=1)


def plot_multiclass_roc(y_true, y_score, n_classes, class_names=None):
    # Binarize the output
    y_test_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        label_name = f"Class {i}" if class_names is None else class_names[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {label_name} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


def print_and_plot_extended_metrics(y_true, y_pred, y_score=None, class_names=None):
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curves plot (if probabilities provided)
    if y_score is not None and len(set(y_true)) > 1:
        n_classes = len(set(y_true))
        plot_multiclass_roc(y_true, y_score, n_classes, class_names)


# Replace with your actual class names
class_names = ["Healthy", "Lame1", "Lame2"]

# Call the evaluation function
print_and_plot_extended_metrics(y_test, pred_classes, pred_probs, class_names)
