import tensorflow as tf
from collections import Counter

filtered_tfrecord = "train_1014.tfrecord"

def verify_all_labels(file_path):
    all_label_stats = Counter()
    total_count = 0
    multi_label_videos = 0

    for raw_record in tf.data.TFRecordDataset(file_path):
        context_features = {
            'labels': tf.io.VarLenFeature(tf.int64)
        }

        context, _ = tf.io.parse_single_sequence_example(
            raw_record, context_features=context_features, sequence_features={}
        )
        
        labels = tf.sparse.to_dense(context['labels']).numpy().tolist()
        
        all_label_stats.update(labels)
        total_count += 1
        if len(labels) > 1:
            multi_label_videos += 1

    print(f"1. amount: {total_count}")
    print(f"2. with multi labels: {multi_label_videos}")
    print(f"3. all labels:")
    for lid, count in all_label_stats.items():
        print(f"   label {lid}: appear {count} times")

if __name__ == "__main__":
    verify_all_labels(filtered_tfrecord)