# OctMem-Agent with sparse block memory

The original OctMem-Agent reader now receives one new RGB-D frame and its camera pose per step. Current-frame pooling uses 5 m cells within 50 m of the camera and 25 m blocks outside that range. Each episode has one sparse index `(block_x, block_y, block_z, slot)`. Slot zero is the block's shared default; slots 1 through 125 address 5 m cells and are stored only where observed.

Repeated writes to a slot average the old and current features equally. The first write to a fine slot averages the current fine feature with the block default from the previous step, if it exists. Fine writes leave that default and other fine slots unchanged. The default contributes only to the feature, not the fine slot's centroid or observation count.

The reader emits each stored slot once and recomputes the original additive relative position embedding from its world centroid. Scene queries see all stored tokens; exploration queries see only coarse tokens more than 40 m from the current camera. The block has five cells per axis and is not a recursive eight-child octree. Runtime memory is not serialized as model weights.

From the repository root, train with `bash train_octmem_agent.sh` (one-step TBPTT by default). Use `python -m utils.merge_lora` to export the adapter and `bash eval.sh` for current-frame evaluation. `--memory_voxel_size` controls the fine grid at training time and is restored from the checkpoint configuration. The block edge is fixed at 25 m and must be an integer multiple of the fine cell size.
