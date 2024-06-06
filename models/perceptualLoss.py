import torchvision.models as models
import torch

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = torch.nn.Sequential()
        model = model.cuda()
        for i ,layer in enumerate(list(cnn)):
            model.add_module(str(i) ,layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):

        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        #f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real)
        return loss

def init_loss():
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    return content_loss
