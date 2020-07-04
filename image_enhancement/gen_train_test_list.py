import os
from tqdm import tqdm

color_img_path = 'images/'
label_img_path = 'res/'

imgs = os.listdir(color_img_path)
limgs = os.listdir(label_img_path)

imgs.sort(key=str.lower)
limgs.sort(key=str.lower)

f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')

pd = tqdm(total=70816)
i = 0
for img, limg in zip(imgs, limgs):
    pd.update(1)
    if i >= 60816:
        f2.write('images/' + img + ' ' + 'res/' + limg + '\n')
    else:
        f1.write('images/' + img + ' ' + 'res/' + limg + '\n')
    i += 1
pd.close()
f1.close()
f2.close()
