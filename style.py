import torch
from torch.autograd import Variable
from model import vgg
from util import saveImg
import numpy as np

def features(model, x, featureLayer = [1,2,3,4, 5,], start = 2, end = 33, **kwargs):
    layers = list(model.modules())[start: end]
    r_features = []
    for i,layer in enumerate(layers):
        x = layer(x)
        if i in featureLayer:
            r_features.append(x)
    return r_features



def create(model, img, style_img, step = 2000, weight = 0.2, lr = 0.01, saveName='tmp.jpg', **kwargs):
    target = img.clone()
    target = Variable(target,  requires_grad = True)
#    optimizer = torch.optim.Adam([target], lr = lr, betas=[0.5, 0.999])
    optimizer = torch.optim.RMSprop([target], lr = lr)
    for tmp_step in range(step):
        target_features = features(model, target, **kwargs)
        content_features = features(model, Variable(img), **kwargs)
        style_features = features(model, Variable(style_img), **kwargs)
        
        content_loss = Variable(torch.zeros([1]))
        style_loss = Variable(torch.zeros([1]))
        
        for target_f, content_f, style_f in zip(target_features, content_features, style_features):
            
            content_loss += torch.mean((target_f - content_f)**2)
            
            tn, tc, th, tw = target_f.size()
            sn, sc, sh, sw = style_f.size()
            target_f = target_f.view(tc * tn, th * tw)
            style_f= style_f.view(sc * sn, sh * sw)
#            print('target shape:',target_f.shape)
#            print('style shape:',style_f.shape)
            target_f_gram = torch.mm(target_f, target_f.t()) / (tn * tc * th * tw) 
            style_f_gram = torch.mm(style_f, style_f.t()) / (sn * sc * sh * sw)
            style_loss += torch.mean((target_f_gram - style_f_gram)**2) 
        
        loss = content_loss + style_loss * weight
        optimizer.zero_grad()
        loss.backward()
#        ratio = np.abs(target.grad.data.cpu().numpy()).mean()
#        learning_rate_use = 0.01 / ratio
#        target.data.add_(target.grad.data * learning_rate_use)
        optimizer.step()
#        return target.data
        if tmp_step % 10 == 0:
            saveImg(target.data[0].numpy(),saveName)
            print('[{}/{}]: loss:{}'.format(tmp_step, step, loss.data.numpy()))
        
    return target.data.numpy()
