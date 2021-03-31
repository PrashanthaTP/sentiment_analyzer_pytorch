
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
def display_confusion_matrix(preds,labels):
    targets = ('positive','negative')
    print('plotting confusion matrix...')
    print(preds[:10].astype('int32'),labels[:10].astype('int32'),)
    cm = confusion_matrix(labels.astype('int32'),preds.astype('int32'))
    print(cm)
    sn.heatmap(cm, annot=True,fmt='d',xticklabels=targets,yticklabels=targets,cmap='Blues')
    plt.title('Confusion Matrix',fontweight='bold')
    plt.xlabel('Predicted',fontweight='bold')
    plt.ylabel('Actual',fontweight='bold')
    plt.show()
