# identify-code
simple identification of verification code
包含3个部分：1、gen_image.py 生成验证码图片以及验证码对应的真实字符形成的csv文件
            2、gen_tfrecords.py 使用tensorflow读取图片和csv文件，并写入tfrecords文件
            3、captcha_train.py 只使用一个全连接层的神经网络对提供的图片特征值进行训练形成模型，最终预测准确率在97%左右。
