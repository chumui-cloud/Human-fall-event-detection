from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def evaluation_metrics(y_test, y_pred):
  y_t = np.argmax(y_test,axis=1)
  y_p = np.argmax(y_pred,axis=1)
  accuracy= accuracy_score(y_t, y_p)

  cf= confusion_matrix(y_t, y_p)

  precision= precision_score(y_t, y_p)

  recall = recall_score(y_t, y_p)

  fscore= f1_score(y_t, y_p)

  roc_auc= roc_auc_score(y_test, y_pred)
  
  return accuracy, cf, precision, recall, fscore, roc_auc

def cf_plot(y_test, y_pred):
  y_test = np.argmax(y_test,axis=1)
  y_pred = np.argmax(y_pred,axis=1)
  cf= confusion_matrix(y_test, y_pred)
  ax = sns.heatmap(cf, annot=True, cmap='Blues')

  ax.set_title('Confusion Matrix\n\n');
  ax.set_xlabel('\nPredicted Labels')
  ax.set_ylabel('True Labels');
  ax.xaxis.set_ticklabels(['Non Fall','Fall'])
  ax.yaxis.set_ticklabels(['Non Fall','Fall'])
  plt.show() 

def roc_auc_plot(y_test, y_pred):
  
  y_test = y_test[:][1]
  y_pred = y_pred[:][1]
  fpr, tpr, threshold = roc_curve(y_test, y_pred)
  roc_auc = auc(fpr, tpr)
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

def plot_history(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train Acc', 'Val Acc'], loc='lower right')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train loss', 'Val loss'], loc='upper right')
  plt.show()