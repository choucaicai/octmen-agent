from typing import Callable, Tuple

import tensorflow as tf


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image


def resize_depth_image(depth_image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes a depth image using bilinear interpolation. Expects & returns float32 in arbitrary range."""
    assert depth_image.dtype == tf.float32
    if len(depth_image.shape) < 3:
        depth_image = tf.image.resize(
            depth_image[..., None], size, method="bilinear", antialias=True
        )[..., 0]
    else:
        depth_image = tf.image.resize(
            depth_image, size, method="bilinear", antialias=True
        )
    return depth_image


def read_resize_encode_image(path: str, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(data)
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return tf.io.encode_jpeg(image, quality=95)

# def vmap(fn: Callable) -> Callable:
#     """
#     Vmap a function over the first dimension of a tensor (or nested structure of tensors). This
#     version does NOT parallelize the function; however, it fuses the function calls in a way that
#     appears to be more performant than tf.map_fn or tf.vectorized_map (when falling back to
#     while_loop) for certain situations.

#     Requires the first dimension of the input to be statically known.
#     """
#     def wrapped(structure):
#         res =  tf.nest.map_structure(
#             lambda *x: tf.stack(x),
#             *[
#                 fn(tf.nest.pack_sequence_as(structure, x))
#                 for x in zip(*map(tf.unstack, tf.nest.flatten(structure)))
#             ],
#         )
#         return res
#     return wrapped

def vmap(fn: Callable) -> Callable:
    def wrapped(structure):
        key_list = list(structure.keys())

        flattened_structure = tf.nest.flatten(structure)

        # print(f'flattened_structure:{key_list}',)
        # print('flattened_structure:',len(flattened_structure),flattened_structure[0].shape)
        # print('flattened_structure:',flattened_structure)
        # unstacked_tensors = list(map(tf.unstack, flattened_structure))
        # 用 tf.unstack 或手动索引
        first_tensor = flattened_structure[0]
        if isinstance(first_tensor, tf.RaggedTensor):
            batch_size = first_tensor.nrows()
        else:
            batch_size = tf.shape(first_tensor)[0]
        # tf.print('batch_size:', batch_size)
        
        def process_single_sample(i):
            # 提取第 i 个样本的所有字段
            single_sample_flat = [t[i] for t in flattened_structure]
            single_sample = tf.nest.pack_sequence_as(structure, single_sample_flat)
            # tf.print('res:', single_sample)
            result = fn(single_sample)
            # tf.print('fn:', result)
            return result
        
        def make_spec(tensor):
            if isinstance(tensor, tf.RaggedTensor):
                # RaggedTensor: 保留嵌套层级
                inner_shape = tensor.shape[1:]  # 去掉 batch 维度
                return tf.TensorSpec(shape=inner_shape, dtype=tensor.dtype)
            else:
                # 普通 Tensor: shape 去掉第一维
                return tf.TensorSpec(
                    shape=tensor.shape[1:] if tensor.shape.ndims  > 0 else (),
                    dtype=tensor.dtype
                )
        sample_0 = tf.nest.map_structure(lambda t: t[0], structure)
        sample_output = fn(sample_0)  # 执行一次
        output_signature = tf.nest.map_structure(
            lambda t: tf.TensorSpec(shape=t.shape, dtype=t.dtype),
            sample_output
        )
        # 用 tf.map_fn 批量处理所有样本
        transformed_samples = tf.map_fn(
            process_single_sample,
            tf.range(batch_size),
            fn_output_signature=output_signature,
        )
        # print('transformed_samples:',transformed_samples)
        # result = tf.nest.map_structure(
        #     lambda *x: tf.stack(x),  # 堆叠函数：接收多个张量，返回堆叠后的张量
        #     *transformed_samples      # 展开所有样本的结果
        # )
        result = transformed_samples
        return result
    
    return wrapped

def parallel_vmap(fn: Callable, num_parallel_calls=tf.data.AUTOTUNE) -> Callable:
    """
    Vmap a function over the first dimension of a tensor (or nested structure of tensors). This
    version attempts to parallelize the function using the tf.data API. I found this to be more
    performant than tf.map_fn or tf.vectorized_map (when falling back to while_loop), but the batch
    call appears to add significant overhead that may make it slower for some situations.
    """

    def wrapped(structure):
        return (
            tf.data.Dataset.from_tensor_slices(structure)
            .map(fn, deterministic=True, num_parallel_calls=num_parallel_calls)
            .batch(
                tf.cast(tf.shape(tf.nest.flatten(structure)[0])[0], tf.int64),
            )
            .get_single_element()
        )

    return wrapped
