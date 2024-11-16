import torch
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

def reconstruct_observation(flat_observation, device = None):
    # Assuming the first part of the flat array is the image
    if type(flat_observation) is not torch.Tensor:
        image = flat_observation[:, 16:]  # 9*9*3 = 243
        image = torch.tensor(image.reshape((flat_observation.shape[0], 13, 13, 4)))
        # The remaining part is the action mask
        action_mask = torch.tensor(flat_observation[:, :8])
        advice_mask = torch.tensor(flat_observation[:,8:16])
    else:
        image = flat_observation[:, 16:].clone().detach()  # 9*9*3 = 243
        image = image.reshape((flat_observation.shape[0], 13, 13, 4))
        # The remaining part is the action mask
        action_mask = flat_observation[:, :8].clone().detach()
        advice_mask = flat_observation[:, 8:16].clone().detach()
    return {"obs":{'image': image, 'action_mask': action_mask, 'advice_mask':advice_mask}}


def manage_batch(sample_batch: SampleBatch, new_sample: SampleBatch, batch_size):
    if sample_batch is None:
        sample_batch = new_sample
    else:
        sample_batch = sample_batch.concat(new_sample)
    cur_batch_size = sample_batch[SampleBatch.CUR_OBS].shape[0]
    num_to_remove = max(0, cur_batch_size-batch_size)
    if num_to_remove > 0:
        for key in sample_batch.keys():
            sample_batch[key] = sample_batch[key][num_to_remove:]
    return sample_batch


def sample_from_batch(sample_batch: SampleBatch, k, device):
    total_samples = sample_batch[SampleBatch.CUR_OBS].shape[0]
    
    # Ensure k is not greater than the total number of samples
    k = min(k, total_samples)
    
    # Randomly select k unique indices
    selected_indices = np.random.choice(total_samples, k, replace=False)
    
    # Extract samples for each key based on the selected indices
    sampled_batch = {}
    for key in sample_batch.keys():
        sampled_batch[key] = sample_batch[key][selected_indices]
    
    return SampleBatch(sampled_batch).to_device(device)