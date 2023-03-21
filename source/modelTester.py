import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


model_path='models/InceptionV3_320x320/Mymodel320.h5'
testimage_path = 'testimage'
result_path='results/testresult/'
width=320
height=320

#################LOAD MODEL###########################
model = tf.keras.models.load_model(model_path)
inputs = tf.keras.Input(shape=(width,height,3))
x = model(inputs,training=False)
model = tf.keras.Model(inputs,x)
model.summary()


################LOAD SET OF TEST IMAGES#################
test_gen = ImageDataGenerator(rescale=1./255)
test_set = test_gen.flow_from_directory(testimage_path,
                                        target_size=(width,height),
                                        class_mode=None)


#################PREDICTION############################

test_item = next(test_set)
test_list = list(test_item)


count = 0
for image in test_list:
    figure = plt.figure()
    count = count+1
    img_array = tf.expand_dims(image,0)
    prediction = model.predict(img_array)
    print(prediction[0])
    if prediction[0] < 0.5:
        output = 'BULLONE: '+ str(prediction[0])
    else:
        output = 'NO BULLONE' + str(prediction[0])

    plt.figtext(0.02,0.02,output,c='red',fontsize = 'large',fontstyle = 'italic')
    plt.imshow(image)
    plt.savefig(result_path+str(count)+'.jpg')
    plt.close(figure)








