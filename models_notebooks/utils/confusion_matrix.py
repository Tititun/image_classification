import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(model, test_ds, class_names):
    predictions = np.array([])
    labels = np.array([])
    for x, y in test_ds:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=0),
                                                             axis=1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    cm = confusion_matrix(labels, predictions, normalize="true")
    cm = np.round(cm, 2) * 100
    fig, ax = plt.subplots(figsize=(35, 35))
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=class_names)
    cmd.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.show()

