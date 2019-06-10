import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def dump_log(args):
    results = []
    with open(args.input, 'r') as f:
        iou = 0
        obj = 0
        recall = 0
        boxes = 0
        cnt = 0
        for line in f:
            m = re.match('^Region Avg IOU: ([\.0-9]+), Class: [\.0-9]+, Obj: ([\.0-9]+), No Obj: [\.0-9]+, Avg Recall: ([\.0-9]+),  count: (\d+)', line)
            if m is not None:
                iou = iou + float(m.group(1))
                obj = obj + float(m.group(2))
                recall = recall + float(m.group(3))
                boxes = boxes + int(round(float(m.group(3))*float(m.group(4))+0.1))
                cnt = cnt+1
                #print(int(round(float(m.group(3))*float(m.group(4))+0.1)))
            else:
                m = re.match('^(\d+): \d+\.\d+, ([0-9\.]+) avg, (0\.\d+) rate', line)
                if m is not None:
                    #print('iou: {}, obj: {}, boxes: {}'.format(iou/cnt, obj/cnt, boxes))
                    #print('loss: {}, lr: {}'.format(m.group(2), m.group(3)))
                    results.append([float(m.group(2)), float(m.group(3)), iou/cnt, obj/cnt, recall/cnt, boxes])
                    iou = 0
                    obj = 0
                    recall = 0
                    boxes = 0
                    cnt = 0
    df = pd.DataFrame(results, columns=['loss', 'lr', 'iou', 'obj', 'recall', 'boxes'])
#    print(df.dtypes)
    total = len(df['loss'])
    step = int(total / 200)
    if step > 5:
        out = []
        for i in range(int(total/step)):
            l = df['loss'][i*step:(i+1)*step]
            iou = df['iou'][i*step:(i+1)*step]
            obj = df['obj'][i*step:(i+1)*step]
            rec = df['recall'][i*step:(i+1)*step]
            lr = df['lr'][i*step:(i+1)*step]
            out.append([np.average(l), np.average(lr), np.average(iou), np.average(obj), np.average(rec)])
        df = pd.DataFrame(out, columns=['loss', 'lr', 'iou', 'obj', 'recall'])
    else:
        step = 1
    
    plt.rcParams["figure.figsize"] = (8,15)
    plt.subplot(311)
    plt.plot(df['loss'], label='Loss')
    plt.xlabel("Steps(x{})".format(step))
    plt.ylabel("Loss")
    plt.title('Learning Curve')
    plt.tight_layout()

    plt.subplot(312)
    plt.plot(df['lr'], label='learning rate')
    plt.xlabel("Steps(x{})".format(step))
    plt.ylabel("Learning Rate")
    plt.title('Learning Curve')
    plt.tight_layout()
    
    plt.subplot(313)
    plt.plot(df['iou'], label='iou')
    plt.plot(df['obj'], label='objectness')
    plt.plot(df['recall'], label='recall')
    plt.xlabel("Steps(x{})".format(step))
    plt.ylabel("Percentage")
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.show()
    plt.savefig('trainning_result.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file name')
    
    args = parser.parse_args()
    dump_log(args)