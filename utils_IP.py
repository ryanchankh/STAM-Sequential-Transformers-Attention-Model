"""Operations for processing patches."""

import copy
import torch
import torch.nn.functional as F



def sample_random_history(num_samples, num_queries_total, max_queries):
    """Sample indices in a uniformly random manner. 

    Arguments: 
        num_queries_total: total number of queries available to choose
        max_queries: maximum number of queries to sample

    Return:
        mask_indices
    """
    num_queries = torch.randint(low=0, high=max_queries, size=(num_samples, ))
    indices = torch.zeros(num_samples, num_queries_total)
    for code_ind, num in enumerate(num_queries):
        if num == 0:
            continue
        random_history = torch.multinomial(torch.ones(indices.size(1)), num, replacement=False)
        indices[code_ind, random_history.flatten()] = 1.
    return indices

def sample_random_history_fixed(num_samples, num_queries_total, max_queries):
    """Sample indices in a uniformly random manner. 

    Arguments: 
        num_queries_total: total number of queries available to choose
        max_queries: maximum number of queries to sample

    Return:
        mask_indices
    """
    num = torch.randint(low=0, high=max_queries, size=(1, ))[0]
    indices_onehot = torch.zeros(num_samples, num_queries_total)
    indices = []
    for code_ind in range(num_samples):
        random_history = torch.multinomial(torch.ones(indices.size(1)), num, replacement=False)
        indices_onehot[code_ind, random_history.flatten()] = 1.
        indices.append(random_history.flatten())
    return indices_onehot, torch.stack(indices_onehot)


def mask_image_from_indices_mnist(images, query_indices, patch_size):
    """Obtain masked images, with patch unveiled at indices.

    Arguments:
        images: full images
        mask_indices: indices, with 0 is hide and 1 is keep
        patch_size: size of the patch to unveil

    Returns:
        masked image, same size as x. 
    
    """
    device = images.device
    N, C, H, W = images.shape
    
    
    query_vec = query_indices.view(N, 1, H-patch_size+1, W-patch_size+1)
    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    query_mask = F.conv2d(query_vec, kernel, stride=1, padding=patch_size-1, bias=None)
    return query_mask.clamp_(min=-1., max=1.) * images

def update_masked_image_mnist(masked_image, images, query_vec, patch_size):
    """Update mask image by unveiling the patch that was selected by query vec.
    This method is differentiable. 
    
    Arguments:
        masked_images: masked image by history
        images: full images
        query_vec: vector of queried indices, (N, |Q|)
        mask_indices: indices, with 0 is hide and 1 is keep
        patch_size: size of the patch to unveil

    Returns:
        masked image with newly added query image, same size as image. 
    
    """
    device = images.device
    N, C, H, W = images.shape
    
    # make a copy of masked image
    masked_image = copy.deepcopy(masked_image)

    # create query mask
    query_vec = query_vec.view(N, 1, H-patch_size+1, W-patch_size+1)
    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    query_mask = F.conv2d(query_vec, kernel, stride=1, padding=patch_size-1, bias=None)
  
    modified_history = (masked_image + query_mask * images).clamp_(min=-1., max=1.)
    return modified_history



def onehot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def hardargmax(logits, dim=1):
    _, d = logits.shape
    y = torch.argmax(logits, dim=dim)
    return onehot(y, d)
