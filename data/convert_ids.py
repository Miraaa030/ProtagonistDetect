import tensorflow as tf
import requests
import re
import time
import os

# 配置
TEST_RECORD = "test_1014.tfrecord"
OUTPUT_IDS_TXT = "test_ids.txt"
OUTPUT_LINKS_TXT = "test_real_links.txt"

def get_real_youtube_url(short_id):
    prefix = short_id[:2]
    url = f"http://data.yt8m.org/2/j/i/{prefix}/{short_id}.js"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            match = re.search(r'i\(".+?","(.+?)"\);', response.text)
            if match:
                yt_id = match.group(1)
                return f"https://www.youtube.com/watch?v={yt_id}"
    except:
        pass
    return None

def process_test_set():
    print(f"{TEST_RECORD}")
    
    short_ids = []
    dataset = tf.data.TFRecordDataset(TEST_RECORD)
    for raw_record in dataset:
        context, _ = tf.io.parse_single_sequence_example(
            raw_record, 
            context_features={'id': tf.io.FixedLenFeature([], tf.string)},
            sequence_features={}
        )
        short_ids.append(context['id'].numpy().decode('utf-8'))
    
    results_with_links = []
    
    for i, sid in enumerate(short_ids):
        real_url = get_real_youtube_url(sid)
        if real_url:
            results_with_links.append(f"{sid}\t{real_url}")
        else:
            results_with_links.append(f"{sid}\t[lost]")
        
        if (i + 1) % 10 == 0:
            time.sleep(1)

    with open(OUTPUT_IDS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(short_ids))
    
    with open(OUTPUT_LINKS_TXT, "w", encoding="utf-8") as f:
        f.write("Short_ID\tReal_YouTube_Link\n")
        f.write("-" * 50 + "\n")
        f.write("\n".join(results_with_links))

    print("\n" + "="*30)
    print(f"done")


if __name__ == "__main__":
    process_test_set()