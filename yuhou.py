import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 加载图像并调整大小
def load_images(data_dir, img_size): #从指定目录加载图像文件，并将它们转换成统一大小
    images = []
    labels = []
    for folder in os.listdir(data_dir): #遍历指定路径下的文件夹，其中 os.listdir(data_dir) 返回指定目录中所有文件和文件夹的名称列表
        for file in os.listdir(os.path.join(data_dir, folder)):
            img_path = os.path.join(data_dir, folder, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(folder)
    return np.array(images), np.array(labels)

# 构建模型
def create_model(input_shape, num_classes): #创建神经网络模型。函数接受输入数据的形状 input_shape 和分类数量 num_classes 作为参数
    model = Sequential() #将各个神经网络层按照顺序逐层叠加起来，构成一个“线性”模型
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) #添加了一个卷积层 Conv2D 到模型中 (3,3是滤波器大小)
    #接受输入张量（特征图），尺寸为 input_shape；
    #将每个滤波器应用于输入张量；
    #对每个输出结果应用 ReLU 非线性激活;
    #输出包括32张空间特征图通道
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    #optimizer=Adam() 指定使用 Adam 优化算法；
    #loss='categorical_crossentropy' 表示采用交叉熵作为损失函数，适合多分类问题；
    #metrics=['accuracy'] 说明度量模型性能时以准确率作为衡量标准
    return model

# 主程序
def main():
    data_dir = r'F:\code_test\data'
    img_size = (150, 150)#这是图片的大小根据自己图片修改
    num_classes = 2
    batch_size = 4
    epochs = 50

    # 加载图像数据
    images, labels = load_images(data_dir, img_size)

    # 数据预处理
    images = images.astype('float32') / 255.0
    labels = (labels == 'cancer').astype(int)
    labels = to_categorical(labels, num_classes)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 数据增强
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    datagen.fit(X_train)

    # 创建模型
    model = create_model((img_size[0], img_size[1], 3), num_classes)

    # 训练模型
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs)

    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    # 绘制训练过程的准确率和损失曲线
    def plot_training_history(history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    plot_training_history(history)

if __name__ == '__main__':
    main()