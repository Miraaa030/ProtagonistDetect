"""
data layer
"""
import tensorflow as tf

RGB_DIM    = 1024
AUDIO_DIM  = 128
# Videos longer than this length are truncated
MAX_FRAMES = 300



def parse_sequence_example(example_proto):
    """
    Parse a single SequenceExample
    rgb: [T, 1024]  float32, 0~1
    audio: [T, 128] float32, 0~1
    length: scalar  int32
    video_id: scalar    string
    """
    
    context_features = {
        'id':     tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
    }

    sequence_features = {
        'rgb':   tf.io.FixedLenSequenceFeature([], tf.string),
        'audio': tf.io.FixedLenSequenceFeature([], tf.string),
    }

    context, sequence = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    
    rgb = tf.map_fn(
        lambda x: tf.cast(tf.io.decode_raw(x, tf.uint8), tf.float32) / 255.0,
        sequence['rgb'],
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),
    )
    rgb = rgb.to_tensor(default_value=0.0) 
    rgb = tf.ensure_shape(rgb, [None, RGB_DIM])

    audio = tf.map_fn(
        lambda x: tf.cast(tf.io.decode_raw(x, tf.uint8), tf.float32) / 255.0,
        sequence['audio'],
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),
    )
    audio = audio.to_tensor(default_value=0.0)
    audio = tf.ensure_shape(audio, [None, AUDIO_DIM])

    # Truncate
    length_raw = tf.shape(rgb)[0]
    rgb        = rgb[:MAX_FRAMES]
    audio      = audio[:MAX_FRAMES]
    length     = tf.cast(tf.minimum(length_raw, MAX_FRAMES), tf.int32)

    return rgb, audio, length, context['id']



def _add_mask_and_reformat(rgb, audio, length, video_id):
    """
    Arrange the tuple output of padded_batch into a dict with mask
    """
    t_max = tf.shape(rgb)[1]
    mask  = tf.sequence_mask(length, maxlen=t_max, dtype=tf.bool)   # [B, T]

    return {
        'video_id': video_id,
        'rgb':      rgb,
        'audio':    audio, 
        'length':   length, 
        'mask':     mask,
    }



class YT8MLoader:
    """
    YT8M Frame-level SequenceExample dataset loader
    """

    def __init__(
        self,
        tfrecord_path,
        batch_size    : int  = 32,
        shuffle       : bool = False,
        shuffle_buffer: int  = 1000,
        num_parallel          = tf.data.AUTOTUNE,
    ):
        self.tfrecord_path  = tfrecord_path if isinstance(tfrecord_path, list) \
                              else [tfrecord_path]
        self.batch_size     = batch_size
        self.shuffle        = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.num_parallel   = num_parallel

    def get_dataset(self) -> tf.data.Dataset:
        """
        return tf.data.Dataset
        'video_id' : [B]
        'rgb'      : [B, T, 1024]  with padding
        'audio'    : [B, T, 128]   with padding
        'length'   : [B]
        'mask'     : [B, T] 
        """
        dataset = tf.data.TFRecordDataset(
            self.tfrecord_path,
            num_parallel_reads=self.num_parallel,
        )

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer, seed=42)

        # parse SequenceExample
        dataset = dataset.map(
            parse_sequence_example,
            num_parallel_calls=self.num_parallel,
        )

        # padded_batch：padding to the maximum length within the batch
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                [None, RGB_DIM], 
                [None, AUDIO_DIM], 
                [], 
                [], 
            ),
            padding_values=(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0,   dtype=tf.int32),
                tf.constant(b'', dtype=tf.string),
            ),
            drop_remainder=False,
        )

        # add mask and output dict
        dataset = dataset.map(
            _add_mask_and_reformat,
            num_parallel_calls=self.num_parallel,
        )

        return dataset.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    import sys
    import numpy as np

    tfrecord = sys.argv[1] if len(sys.argv) > 1 else 'data/val_1014.tfrecord'
    print(f'[loader] read: {tfrecord}\n')

    loader = YT8MLoader(tfrecord, batch_size=4, shuffle=False)
    ds     = loader.get_dataset()

    for batch in ds.take(1):
        B = batch['rgb'].shape[0]
        print('Batch:')
        print(f"  video_id : {batch['video_id'].shape}")
        print(f"  rgb      : {batch['rgb'].shape}   dtype={batch['rgb'].dtype}")
        print(f"  audio    : {batch['audio'].shape} dtype={batch['audio'].dtype}")
        print(f"  length   : {batch['length'].numpy()}")
        print(f"  mask     : {batch['mask'].shape}")

        print('\nrange:')
        mask0  = batch['mask'][0].numpy()
        rgb0   = batch['rgb'][0].numpy()[mask0]
        audio0 = batch['audio'][0].numpy()[mask0]

        print(f'  rgb   min={rgb0.min():.4f}  max={rgb0.max():.4f}')
        print(f'  audio min={audio0.min():.4f}  max={audio0.max():.4f}')
        print(f'  real lenth: {batch["length"][0].numpy()} 秒')
        print(f'  padding frames  : {batch["rgb"].shape[1] - batch["length"][0].numpy()}')

        print(f'\n  video_id[0]: {batch["video_id"][0].numpy().decode()}')

