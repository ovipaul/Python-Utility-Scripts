import numpy as np
from sklearn.metrics import confusion_matrix
from img_multi_to_bin import img_multi_to_bin
from numba import jit

@jit(nopython=True)
def multiToBinImg(tarImg, predImg):
    numClass = 1
    for i in range(numClass):
       
       tarVal = img_multi_to_bin(tarImg,i)
       predVal = img_multi_to_bin(predImg,i)   

    return [tarVal, predVal]

def confusionMatrix(tarVal, predVal):
       
    y_true = (tarVal==1)
    y_pred = (predVal==1)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    tn, fp, fn, tp=confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return [tn, fp, fn, tp]

def precision(tp, fp):
    if tp == 0:
        return 1
    elif tp + fp == 0:
        return 1
    else:
        return tp / (tp + fp)


def recall(tp, fn):
    if tp == 0:
        return 1
    elif tp + fn == 0:
        return 1
    else:
        return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return 1
    else:
        return (tp + tn) / (tp + tn + fp + fn)
        
# def iou(target, predict):
#     intersection = np.logical_and(target, predict)
#     union = np.logical_or(target, predict)
#     try:
#         iou_score = np.sum(intersection) / np.sum(union)
#         if iou_score is None:
#             iou_score =0
#     except ZeroDivisionError:
#         print("Division by Zero")
#     return iou_score

def iou(target, predict):
    intersection = np.logical_and(target, predict)
    union = np.logical_or(target, predict)
    iou_score = np.sum(intersection) / np.sum(union)
    if iou_score is None:
        print("null")
        return 0
    else:
        return iou_score