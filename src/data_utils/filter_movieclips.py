import tensorflow as tf
import os

# 配置
MOVIE_LABEL_ID = 1014
INPUT_DIR = "2/frame/train" 
OUTPUT_FEATURES = "movieclips_frame_features.tfrecord"
OUTPUT_IDS = "movieclips_ids.txt"


feature_desc = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "labels": tf.io.VarLenFeature(tf.int64),
    "rgb": tf.io.VarLenFeature(tf.string),  
    "audio": tf.io.VarLenFeature(tf.string), 
}

def parse_example(proto):
    return tf.io.parse_single_example(proto, feature_desc)

def has_movieclips(example):
    labels = tf.sparse.to_dense(example["labels"])
    return tf.reduce_any(tf.equal(labels, MOVIE_LABEL_ID))

movie_ids = set()
writer = tf.io.TFRecordWriter(OUTPUT_FEATURES)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".tfrecord"):
        continue
    path = os.path.join(INPUT_DIR, fname)
    print(f"处理: {fname}")
    
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_example).filter(has_movieclips)
    
    for ex in dataset:
        vid = ex["id"].numpy().decode()
        movie_ids.add(vid)
        
        feature = {
            "id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ex["id"].numpy()])),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=tf.sparse.to_dense(ex["labels"]).numpy())),
            "rgb": tf.train.Feature(bytes_list=tf.train.BytesList(value=tf.sparse.to_dense(ex["rgb"]).numpy())),
            "audio": tf.train.Feature(bytes_list=tf.train.BytesList(value=tf.sparse.to_dense(ex["audio"]).numpy())),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())

writer.close()

with open(OUTPUT_IDS, "w") as f:
    for vid in sorted(movie_ids):
        f.write(vid + "\n")

