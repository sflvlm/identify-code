import tensorflow as tf
from captcha.image import ImageCaptcha
import random
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', './pic_data/', '生成图片存放路径')
tf.app.flags.DEFINE_integer('Captcha_size', 4, '每张验证码的字符数量')
tf.app.flags.DEFINE_integer('image_num', 4000, '生成验证码的数量')

# 验证码内容
Captcha_content = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# 生成字符
def random_captcha_text():
    captcha_text = []
    # 生成字符串列表
    for i in range(FLAGS.Captcha_size):
        ch = random.choice(Captcha_content)
        captcha_text.append(ch)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=80, height=30, font_sizes=(21, 25, 28))
    csv_list = []
    for i in range(FLAGS.image_num):
        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)
        image.write(captcha_text, (FLAGS.output_dir + '%s' + '.jpg') % i)
        csv_dict = list((i, captcha_text))
        csv_list.append(csv_dict)
    return csv_list

def main():

    csv_list = gen_captcha_text_and_image()

    f = open(FLAGS.output_dir + 'label.csv', 'w', encoding='utf-8', newline="")
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 4. 写入csv文件内容
    for i in csv_list:
        csv_writer.writerow(i)
    # 5. 关闭文件
    f.close()


if __name__ == '__main__':

    main()
