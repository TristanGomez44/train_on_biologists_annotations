import torch 

import gc 
import time,sys

class RISE(torch.nn.Module):

    def __init__(self,model,nbMasks=4000,batchSize=400,res=7):
        super().__init__()

        nbMasks = nbMasks * (res//7)**4

        print("RISE RES",res,"NB_MASKS",nbMasks,"BATCH_SIZE",batchSize)

        self.nbMasks = nbMasks
        self.batchSize = batchSize
        self.model = model.eval() 
        self.res = res

    def forward(self,x,target=None):

        x_cpu = x
        if target is None:
            ind = torch.argmax(self.model(x_cpu)["pred"][0])
        else:
            ind = target

        if type(ind) is torch.Tensor:
            ind = ind.item()
        
        totalMaskNb = 0
        masks = torch.zeros(self.batchSize,1,self.res,self.res).to(x.device)

        batchNb = 0

        allMasks,allOut = None,None
        out = None

        while totalMaskNb < self.nbMasks:
   
            masks = masks.bernoulli_()
            x_mask = x*torch.nn.functional.interpolate(masks,size=(x.size(-1)))

            resDic = self.model(x_mask.cpu()) 

            out = resDic["pred"][:,ind]
            allMasks = masks.cpu() if allMasks is None else torch.cat((allMasks,masks.cpu()),dim=0)
            allOut =  out.cpu() if allOut is None else torch.cat((allOut,out.cpu()),dim=0)

            totalMaskNb += self.batchSize

            batchNb += 1

        allOut = allOut.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        salMap = (allMasks*allOut).sum(dim=0,keepdim=True)/(allMasks.mean(dim=0,keepdim=True)*totalMaskNb)

        return salMap
