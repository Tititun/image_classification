import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

cm = confusion_matrix(labels, predictions, normalize="true")

