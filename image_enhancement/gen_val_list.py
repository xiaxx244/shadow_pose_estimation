import os
from tqdm import tqdm

color_img_path = ''
#label_img_path = 'segmentation/res/'

imgs = os.listdir(color_img_path)
#limgs = os.listdir(label_img_path)

imgs.sort(key=str.lower)
#limgs.sort(key=str.lower)

f1 = open('val.txt', 'w')
#f2 = open('test.txt', 'w')

pd = tqdm(total=10001)
i = 0
for img in imgs:
    pd.update(1)
    f1.write('val_images/' + img + '\n')
    i += 1
pd.close()
f1.close()
