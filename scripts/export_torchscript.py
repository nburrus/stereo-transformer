from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../') # add relative path
sys.path.append('./') # add relative path

from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor

# %% [markdown]
# ### Define STTR model

# %%
# Default parameters
args = type('', (), {})() # create empty args
args.channel_dim = 128
args.position_encoding='sine1d_rel'
args.num_attn_layers=6
args.nheads=8
args.regression_head='ot'
args.context_adjustment_layer='cal'
args.cal_num_blocks=8
args.cal_feat_dim=16
args.cal_expansion_ratio=4

# %%
model = STTR(args).eval()

# %%
# Load the pretrained model

model_file_name = "../kitti_finetuned_model.pth.tar"
training_set = "kitti"

# model_file_name = "../sceneflow_pretrained_model.pth.tar"
# training_set = "sceneflow"

# model_file_name = "../sttr_light_sceneflow_pretrained_model.pth.tar"
# training_set = "sceneflow_light"

checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict, strict=False)
print("Pre-trained model successfully loaded.")

# %% [markdown]
# ### Read image

# %%
left = np.array(Image.open('../sample_data/KITTI_2015/training/image_2/000046_10.png'))
right = np.array(Image.open('../sample_data/KITTI_2015/training/image_3/000046_10.png'))
disp = np.array(Image.open('../sample_data/KITTI_2015/training/disp_occ_0/000046_10.png')).astype(np.float) / 256.

# %%
# Visualize image
plt.figure(1)
plt.imshow(left)
plt.figure(2)
plt.imshow(right)
plt.figure(3)
plt.imshow(disp)

# %% [markdown]
# Preprocess data for STTR

# %%
# normalize
input_data = {'left': left, 'right':right} # , 'disp':disp}
for k in input_data.keys():
    input_data[k] = normalization(input_data[k])

# %%
# donwsample attention by stride of 3
h, w, _ = left.shape
bs = 1

downsample = 3
col_offset = int(downsample / 2)
row_offset = int(downsample / 2)
sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()

# %%
# build NestedTensor
input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,], sampled_cols=sampled_cols, sampled_rows=sampled_rows)

# %% [markdown]
# ### Inference

# %%
input_data

# %%
# output = model(*input_data)

# %%

for device_name in ["cuda"]:
# for device_name in ["cpu"]:
    for w,h,downsample in [(320, 240, 1), (640, 480, 2), (640, 480, 3), (1280, 720, 3)]:
        left = torch.zeros(1,3,h,w)
        right = torch.zeros(1,3,h,w)

        device = torch.device(device_name)
        model = model.to(device)
        model.eval()

        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,]
        sampled_rows = torch.arange(row_offset, h, downsample)[None,]

        sample_input = (left.to(device), right.to(device), sampled_cols.to(device), sampled_rows.to(device))

        with torch.no_grad():
            traced_module = torch.jit.trace(model, sample_input)
            torch.jit.save(traced_module, f"sttr-{training_set}-{device_name}-{h}x{w}-ds{downsample}.scripted.pt")


    # scripted_module = torch.jit.script(model, sample_input)
    # Fails with annoying key error exceptions.
    # torch.onnx.export(scripted_module,                   # model being run
    #               sample_input,              # model input (or a tuple for multiple inputs)
    #               "test.onnx",               # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=12,          # the ONNX version to export the model to
    #               do_constant_folding=False,  # whether to execute constant folding for optimization
    #               input_names = ['left', 'right', 'sampled_cols', 'sampled_rows'],   # the model's input names
    #               output_names = ['disparity', 'occlusion'])

# %%
# set disparity of occ area to 0
# disp_pred = output['disp_pred'].data.cpu().numpy()[0]
# occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
# disp_pred[occ_pred] = 0.0

# # %%
# # visualize predicted disparity and occlusion map
# plt.figure(4)
# plt.imshow(disp_pred)
# plt.figure(5)
# plt.imshow(occ_pred)

# # %% [markdown]
# # ### Compute metrics

# # %%
# # manually compute occluded region
# occ_mask = compute_left_occ_region(w, disp)

# # visualize the known occluded region
# plt.figure(6)
# plt.imshow(occ_mask)

# # %%
# # compute difference in non-occluded region only
# diff = disp - disp_pred
# diff[occ_mask] = 0.0 # set occ area to be 0.0

# # Note: code for computing the metrics can be found in module/loss.py
# valid_mask = np.logical_and(disp > 0.0, ~occ_mask)

# # find 3 px error
# err_px = (diff > 3).sum()
# total_px = (valid_mask).sum()
# print('3 px error %.3f%%'%(err_px*100.0/total_px))

# # find epe
# err = np.abs(diff[valid_mask]).sum()
# print('EPE %f'%(err * 1.0/ total_px))

# # %%



