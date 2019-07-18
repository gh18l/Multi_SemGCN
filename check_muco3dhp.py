import os
import cv2
from scipy.io import loadmat

for i in range(8):
    for j in range(2):
        basepath = "/home/lgh/data1/multi3Dpose/mpi_inf_3dhp/mpi_inf_3dhp_train_set/S%d/Seq%d/" % (i+1, j+1)
        path = basepath + "imageSequence"
        mat_path = basepath + "annot.mat"
        data = loadmat(mat_path)
        for c in range(13):
            joint2d0 = data['annot2'][c][0].reshape(data['annot2'][c][0].shape[0],28,2)
            img_files = os.listdir(path)
            img_files = sorted([filename for filename in img_files if filename.endswith(".jpg") and int((filename.split('_')[1])) == c],
                            key=lambda d: int((d.split('_')[2].split('.')[0])))
            for ind in range(min(len(img_files), len(joint2d0))):
                img = cv2.imread(os.path.join(path, img_files[ind]))
                joint = joint2d0[ind, :,:]
                for k in range(len(joint)):
                    cv2.circle(img, tuple(joint[k,:].astype("int")), 5, (0,0,255), -1)
                img = cv2.resize(img, (750, 750))
                if not os.path.exists(basepath + "view2djoint"):
                    os.makedirs(basepath + "view2djoint")
                cv2.imwrite(basepath + "view2djoint/img_%d_%06d.jpg" % (c, ind), img)
                print("finish S%d Seq%d camera%d image%06d" % (i,j,c,ind))


