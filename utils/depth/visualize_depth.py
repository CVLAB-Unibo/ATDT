from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gt_path', dest='gt_path',  default=None, help='absolute path to gt')
parser.add_argument('--path_list', dest='path_list', nargs='+', type=str, help='list of absolute path to predicted')
parser.add_argument('--names_list', dest='names_list', nargs='+', type=str, help='list of names', default=None)
parser.add_argument('--scale_pred', dest='scale_pred', type=float, default=1,help='scale pred')
parser.add_argument('--max_depth', dest='max_depth',  help='max_depth', type=float, default=100)
parser.add_argument('--min_depth', dest='min_depth',  help='min_depth', type=float, default=0.001)
parser.add_argument('--cm', dest='cm',  help='cm', type=str, default='jet')
parser.add_argument('--save',  help='save', action='store_true')

args = parser.parse_args()

if args.path_list:
    total=len(args.path_list)
    nrows=round(math.sqrt(total))
    ncols=math.ceil(total/nrows)
    nrows+=1
else:
    nrows=1
    ncols=1

if args.gt_path:
    d=cv2.imread(args.gt_path,cv2.IMREAD_UNCHANGED).astype(np.float32)
    d=d[:,:,0]*256*256 + d[:,:,1]*256 + d[:,:,2]
    d=d/(256*256*256 - 1)*1000
    d[d < args.min_depth] = args.min_depth
    d[d > args.max_depth] = args.max_depth
    
    plt.figure()
    plt.subplot(nrows,ncols,ncols//2+1)
    plt.title("gt")
    plt.imshow(d,cmap=args.cm)

if args.save:
    norm = plt.Normalize(vmin=args.min_depth, vmax= args.max_depth)
    cmap = plt.cm.get_cmap(args.cm)
    d = cmap(norm(d))
    plt.imsave(args.gt_path.replace(".png","_colored.png"),d)

if args.path_list:
    predictions = args.path_list
    names = args.names_list if args.names_list else [str(i) for i in range(len(predictions))]

    for i,p in enumerate(predictions):
        pred_value=np.load(p)
        if len(pred_value.shape) == 4:
            pred_value = np.squeeze(pred_value,axis=0)

        pred_value = pred_value * args.max_depth * args.scale_pred
        pred_value[pred_value < args.min_depth] = args.min_depth
        pred_value[pred_value > args.max_depth] = args.max_depth

        if args.save:
            norm = plt.Normalize(vmin=args.min_depth, vmax= args.max_depth)
            cmap = plt.cm.get_cmap(args.cm)
            d = cmap(norm(pred_value[:,:,0]))
            plt.imsave(p.replace(".npy","_colored.png"),d)
        
        plt.subplot(nrows,ncols,ncols+i+1)
        plt.title(names[i])
        plt.imshow(pred_value[:,:,0],cmap=args.cm)

plt.show()
