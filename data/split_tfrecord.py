import tensorflow as tf
import random

input_file = "val_1014.tfrecord"
output_val = "val_final.tfrecord"
output_test = "test_final.tfrecord"

def split_tfrecord(input_path, val_path, test_path, split_ratio=0.5):
    records = []
    for raw_record in tf.data.TFRecordDataset(input_path):
        records.append(raw_record.numpy())
    
    total_count = len(records)

    random.seed(42) 
    random.shuffle(records)

    split_index = int(total_count * split_ratio)
    val_records = records[split_index:]
    test_records = records[:split_index]

    with tf.io.TFRecordWriter(val_path) as writer:
        for record in val_records:
            writer.write(record)

    with tf.io.TFRecordWriter(test_path) as writer:
        for record in test_records:
            writer.write(record)


if __name__ == "__main__":
    split_tfrecord(input_file, output_val, output_test)