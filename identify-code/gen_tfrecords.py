import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', './pic_data/', '生成图片存放路径')
tf.app.flags.DEFINE_string('letter', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '验证码的字母表')
tf.app.flags.DEFINE_string('tfrecords_dir', './tfrecords/captcha.tfrecords', 'tfrecords文件的存放路径')

# 处理图片
def get_captcha_image():
    # 构造文件名
    filename = []
    for i in range(4000):
        name = str(i) + '.jpg'
        filename.append(name)

    # 构造文件目录名列表
    file_list = [os.path.join(FLAGS.output_dir,name) for name in filename]

    # 构造文件队列
    file_queue = tf.train.string_input_producer(file_list, shuffle=False)

    # 构造阅读器
    reader = tf.WholeFileReader()

    # 包含文件内容及标签, 默认读取一张
    key, value = reader.read(file_queue)

    # 解码成张量
    image = tf.image.decode_jpeg(value)

    image.set_shape([30, 80, 3])

    # 批处理 [4000,30,80,3]
    image_batch = tf.train.batch([image],batch_size=4000,num_threads=2,capacity=4000)

    return image_batch

# 处理CSV文件
def get_captcha_label():
    # 构建文件队列, 注意文件路径为列表['./pic_data/label.csv']
    file_queue = tf.train.string_input_producer(['./pic_data/label.csv'],shuffle=False)

    # 构建文件阅读器对象
    reader = tf.TextLineReader()

    # 提取文件内容,key文件名
    key, value = reader.read(file_queue)

    records = [[1],['None']]
    # 按行解析，返回的是每行的列数据
    num, label = tf.decode_csv(value, record_defaults=records)

    label_batch = tf.train.batch([label],batch_size=4000,num_threads=2,capacity=4000)

    return label_batch

# 将字符串转换为数字
def dealwithlabel(label_str):
    # 构建字符索引{0:'A',1:'B',....,25:'Z'}
    letter_dict = dict(enumerate(list(FLAGS.letter)))

    # 键值对反转
    new_dict = dict(zip(letter_dict.values(),letter_dict.keys()))

    letter_num = []
    for letter in label_str:
        nums = []
        for i in letter.decode('utf-8'):
            num = new_dict[i]
            nums.append(num)
        letter_num.append(nums)
    # 转换为张量
    label = tf.constant(letter_num)
    return label

def write_to_tfrecords(image_batch, label_batch):

    # 转换类型
    label_batch = tf.cast(label_batch,tf.uint8)

    # 建立存储器
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)
    # return image_batch
    # 循环将每一个图片上的数据转换为example协议块，序列化后写入
    for i in range(4000):
        # 取出第i个图片数据的特征值和目标值，转换相应类型，图片的特征值要转换为字符串形式
        image = image_batch[i].eval().tostring()

        label = label_batch[i].eval().tostring()

        # 构造一个样本的example
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        }))

        writer.write(example.SerializeToString())

    writer.close()



if __name__ == '__main__':

    # 获取验证码文件中的图片
    image_batch = get_captcha_image()

    # 获取验证码文件当中的标签数据
    label = get_captcha_label()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        # [b'UWKU' b'FPSM' b'MBCD' ... b'XESH' b'FRHO' b'FXCL']
        label_str = sess.run(label)

        # 处理label,由字符串到数字张量
        label_batch = dealwithlabel(label_str)


        write_to_tfrecords(image_batch,label_batch)

        coord.request_stop()

        coord.join(threads)