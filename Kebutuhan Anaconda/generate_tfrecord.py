
import os
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import namedtuple

# Ganti ini dengan daftar kelas yang sesuai dengan dataset Anda
LABEL_MAP = {
    "Gulma daun sempit": 1,
    "Gulma daun lebar": 2,
    "Gulma teki-tekian": 3  # Tambahkan label lain jika perlu
}

def class_text_to_int(row_label):
    if row_label in LABEL_MAP:
        return LABEL_MAP[row_label]
    else:
        print(f'Error: Unrecognized label {row_label}')
        return None

def split(df, group):
    Data = namedtuple("Data", ["filename", "object"])
    grouped = df.groupby(group)
    return [Data(filename, grouped.get_group(x)) for filename, x in zip(grouped.groups.keys(), grouped.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / row['width'])
        xmaxs.append(row['xmax'] / row['width'])
        ymins.append(row['ymin'] / row['height'])
        ymaxs.append(row['ymax'] / row['height'])
        classes_text.append(row['class'].encode('utf8'))
        class_id = class_text_to_int(row['class'])
        if class_id is not None:
            classes.append(class_id)
        else:
            continue  # Jika label tidak dikenali, lewati data tersebut

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(row['height']),
        'image/width': dataset_util.int64_feature(row['width']),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def main(csv_input, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created TFRecord at {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_input', required=True, help="Path to the CSV input")
    parser.add_argument('--image_dir', required=True, help="Path to the images directory")
    parser.add_argument('--output_path', required=True, help="Path to output TFRecord")
    
    args = parser.parse_args()
    main(args.csv_input, args.image_dir, args.output_path)
