
#Code from https://github.com/tleemann/road_evaluation/blob/main/road/imputations.py

# Implementations of our imputation models.
import torch
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

class BaseImputer():
    def __call__(self, img: torch.Tensor, mask: torch.Tensor)-> torch.Tensor:
        """ Call the Imputation function to fill the masked pixels in an image.
            :param img: original image (C,H,W)-tensor
            :param mask: (H,W)-tensor with a binary mask. 0 indicates pixels absent, 1 indicates pixels present.
            :returns: a (C,H,W) tensor, where the original values are kept, if the mask for the pixels is 1 or imputed otherwise.
            The return tensor is copied to cpu()
        """
        raise NotImplementedError("Please implement an imputation function or use an existing imputor.")

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Call the Imputation function to fill the masked pixels in an image. However, in this version,
            and entire batch of images will be processed, which can results in considerable speedup.
            :param img: B original images (B, C, H, W)-tensor
            :param mask: (B, H, W)-tensor with binary masks. 0 indicates pixels absent, 1 indicates pixels present.
            :returns: a (B, C, H, W) tensor, where the original values are kept, if the mask for the pixels is 1 or imputed otherwise.
            The returned tensor is left on the device that this dataloader is instructed to use (may not be CPU).
        """
        raise NotImplementedError("This imputer does not support the bached interface.")

# Some Imputors
class ChannelMeanImputer(BaseImputer):
    """ Impute by the mean value per channel of the image. """

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        for c in range(len(img)):
            mean_c = img[c,:,:].mean()
            imgsubtensor = img[c,:,:]
            imgsubtensor[mask==0] = mean_c
        return img

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        channel_mean_tensor = img.view(img.shape[0], img.shape[1], -1).mean(axis=2) # [B, C]
        c_shape = channel_mean_tensor.shape
        channel_mean_tensor = channel_mean_tensor.unsqueeze(2).unsqueeze(3).expand(c_shape[0], c_shape[1], img.shape[2], img.shape[3])
        return (channel_mean_tensor * (1.0-mask.unsqueeze(1))) + img*mask.unsqueeze(1)

class ZeroImputer(BaseImputer):
    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        return img*mask.unsqueeze(0)
    
    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        assert img.device == mask.device
        return img*mask.unsqueeze(1)

# Code for infilling.
neighbors_weights = [((1,1), 1/12), ((0,1), 1/6), ((-1,1), 1/12), ((1,-1), 1/12), ((0,-1), 1/6), ((-1,-1), 1/12), ((1,0), 1/6), ((-1,0), 1/6)]

class NoisyLinearImputer(BaseImputer):
    def __init__(self, noise=0.01, weighting=neighbors_weights):
        """	
            Noisy linear imputation.	
            noise: magnitude of noise to add (absolute, set to 0 for no noise)
            weighting: Weights of the neighboring pixels in the computation. 
            List of tuples of (offset, weight)
        """
        self.noise = noise
        self.weighting = weighting
    
    @staticmethod 
    def add_offset_to_indices(indices, offset, mask_shape):
        """ Add the corresponding offset to the indices. Return new indices plus a valid bit-vector. """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]
        #print(cord1.shape, indices.shape)
        valid = ((cord0 < 0) | (cord1 < 0) | (cord0 >= mask_shape[0]) | (cord1 >= mask_shape[1]))
        return ~valid, indices+offset[0]*mask_shape[1]+offset[1]

    @staticmethod 
    def setup_sparse_system(mask, img, neighbors_weights):
        """ Vectorized version to set up the equation system.
            mask: (H, W)-tensor of missing pixels.
            Image: (H, W, C)-tensor of all values.
            Return (N,N)-System matrix, (N,C)-Right hand side for each of the C channels.
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
        #print(imgflat.shape)
        indices = np.argwhere(maskflt==0).flatten() # Indices that are imputed in the flattened mask
        coords_to_vidx= np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices)) # lookup_indices =
        #print(coords_to_vidx[:10])
        #coords_to_vidx = {(idx[0].item(), idx[1].item()): i for i, idx in enumerate(indices)} # Coordinates to variable index
        numEquations = len(indices)
        A = lil_matrix((numEquations, numEquations)) # System matrix
        b = np.zeros((numEquations, img.shape[0]))
        sum_neighbors = np.ones(numEquations) # Sum of weights assigned
        #print("My indices:", indices[:10])
        #print("Num indices: ", len(indices))
        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            #print("Using: ", offset, weight)
            # Sum of the neighbors.
            # Take out outliers
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(indices, offset, mask.shape)
            
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid==1).flatten()
            #print(valid_ids[:10], valid_coords[:10])
            #print("Valid:", valid_ids.shape)
            
            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            #print(has_values_ids[:10], has_values_coords[:10])
            #print("Has Values:", has_values_coords.shape)
            b[has_values_ids, :] -= weight*imgflat[:, has_values_coords].T
            
            # Add weights to the system (left hand side)
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5] # Find coordinates in the system.
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
            
            #print("Has No Values:", has_no_values.shape)
            A[has_no_values_ids, variable_ids] = weight
            
            # Reduce weight for invalid
            #print(np.argwhere(valid==0).flatten()[:10])
            sum_neighbors[np.argwhere(valid==0).flatten()] = sum_neighbors[np.argwhere(valid==0).flatten()] - weight

        A[np.arange(numEquations),np.arange(numEquations)] = -sum_neighbors  
        return A, b

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        """ Our linear inputation scheme. """
        """
        This is the function to do the linear infilling 
        img: original image (C,H,W)-tensor;
        mask: mask; (H,W)-tensor
        """
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        indices_linear = np.argwhere(maskflt==0).flatten() # Indices that need to be imputed.
        # Set up sparse equation system, solve system.
        A, b = NoisyLinearImputer.setup_sparse_system(mask.numpy(), img.numpy(), neighbors_weights)
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        # Fill the values with the solution of the system.
        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise*torch.randn_like(res.t())
            
        return img_infill.reshape_as(img)

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        """ Pseudo implementation of batched interface. """
        res_list = []
        in_device = img.device
        for i in range(len(img)):
            res_list.append(self.__call__(img[i].cpu(), mask[i].cpu()))
        return torch.stack(res_list).to(in_device)

class LinearInterpImputer(BaseImputer):

    def __call__(self, img, xmin,xmax,ymin,ymax ) -> torch.Tensor:

        if ymin == 0:
            y_bottom_row = torch.zeros(1,img.shape[1],1,xmax-xmin+1).to(img.device)
        else:
            y_bottom_row = img[:,:,ymin-1:ymin,xmin:xmax+1]

        if ymax == img.shape[2] - 1:
            y_top_row = torch.zeros(1,img.shape[1],1,xmax-xmin+1).to(img.device)
        else:
            y_top_row = img[:,:,ymax+1:ymax+2,xmin:xmax+1]

        y_rows = torch.cat((y_bottom_row,y_top_row),axis=2)

        y_rows = torch.nn.functional.interpolate(y_rows,(ymax-ymin+1,xmax-xmin+1),mode="bilinear")

        if xmin == 0:
            x_left_col = torch.zeros(1,img.shape[1],ymax-ymin+1,1).to(img.device)
        else:
            x_left_col = img[:,:,ymin:ymax+1,xmin-1:xmin]          
        
        if xmax == img.shape[3] - 1:
            x_right_col = torch.zeros(1,img.shape[1],ymax-ymin+1,1).to(img.device)
        else:
            x_right_col = img[:,:,ymin:ymax+1,xmax+1:xmax+2]

        x_cols = torch.cat((x_left_col,x_right_col),axis=3)
        x_cols = torch.nn.functional.interpolate(x_cols,(ymax-ymin+1,xmax-xmin+1),mode="bilinear")

        patch = (x_cols+y_rows)/2

        img[0,:,ymin:ymax+1,xmin:xmax+1] = patch 

        return img 

def _from_str(imputer_str):
    """ Return a default imputer from a string. """
    if imputer_str == "linear":
        return NoisyLinearImputer()
    elif imputer_str == "fixed":
        return ChannelMeanImputer()
    elif imputer_str == "zero":
        return ZeroImputer()
    elif (imputer_str == "gan" or imputer_str == "gain"):
        raise ValueError("GAIN imputer cannot be created via default, because a pretrained model " +
                "needs to be passed. Please use the explicit constructor of road.imputations.GAINImputer.")
    else:
        raise ValueError("Unknown imputer string. Please use {linear, fixed, zero}.")