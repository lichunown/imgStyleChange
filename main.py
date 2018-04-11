from style import create
from util import readImg
from model import vgg

img = readImg('input.png','./imgs')
style_img = readImg('fg.jpg','./imgs')

#create(vgg, img, style_img,step = 1000, saveName='layer1-5.jpg', featureLayer = [1,2,3,4,5])
#create(vgg, img, style_img,step = 1000, saveName='layer11-15.jpg', featureLayer = [11,12,13,14,15])
#create(vgg, img, style_img,step = 1000, saveName='layer21-25.jpg', featureLayer = [21,22,23,24,25])
#create(vgg, img, style_img,step = 1000, saveName='layerinter1.jpg', featureLayer = [1,5,10,15,20,25])
#create(vgg, img, style_img,step = 800, lr = 0.01, saveName='layerinter2.jpg', featureLayer = [0,2,5,7,10])
#create(vgg, img, style_img,step = 800, lr = 0.01, saveName='layerinter3.jpg', featureLayer = [0,2,5,7,10,12])
#create(vgg, img, style_img,step = 800, lr = 0.01, saveName='layerall.jpg', featureLayer = [0,2,5,7,10,12,14,17,19,21,24,26,28])
#create(vgg, img, style_img,step = 800, lr = 0.01, saveName='layerinter4.jpg', featureLayer = [12,14,17,19,21,])
#create(vgg, img, style_img,step = 800, lr = 0.01, saveName='layerinter5.jpg', featureLayer = [17,19,21,24,26,28])
create(vgg, img, style_img,step = 800, lr = 0.03, saveName='tmp.jpg', featureLayer = [0,2,5,7,10,12])