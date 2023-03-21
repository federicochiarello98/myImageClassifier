import tensorflow as tf
import onnx
import onnxruntime
import cv2 as cv
import numpy as np
import time

##########################IMG SIZE############################################
img_height = 320
img_width = 320

#########################CAMERA INDEX########################################
cam_index = '/dev/video6'

######################LOAD MODEL##############################################

model_dir_path = 'models/InceptionV3_320x320/'
model_name='Mymodel320.onnx'

sess_opt = onnxruntime.SessionOptions()
sess_opt.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
sess_opt.inter_op_num_threads = 8



onnx_session = onnxruntime.InferenceSession(model_dir_path+model_name,sess_opt,providers=["CPUExecutionProvider"])

io_binding = onnx_session.io_binding()

print(onnxruntime.get_device())
onnx_model = onnx.load(model_dir_path+model_name)
onnx.checker.check_model(onnx_model)
input_name = onnx_session.get_inputs()[0].name

#####################PREDICTION#################################################

cap = cv.VideoCapture(cam_index)
if not cap.isOpened():
    print("No camera")
    exit()
count=0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    frame = cv.resize(frame,(img_width,img_height))

    image= cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    image = image/255
    img_array = np.expand_dims(image,0)
    img_array = img_array.astype(np.float32)

    start = time.time()

    prediction = onnx_session.run(None,{input_name:img_array})[0]

    latency = (time.time()-start)

    if prediction[0] < 0.1:
        output = 'BULLONE: ' + str(prediction[0])
        color = (0,255,0)
    else:
        output = 'NO BULLONE: ' + str(prediction[0])
        color = (0,0,255)

    print("{} Inference Time {} ms".format(output,(round(latency*1000,2))))

    cv.putText(frame,output,(20,40),cv.FONT_HERSHEY_SIMPLEX,0.5,color=color)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
