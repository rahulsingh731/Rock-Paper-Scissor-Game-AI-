from torchvision.models import *
from fastai import *
from fastai.vision import *
from fastai.vision.models import *
import cv2

train_directory='/home/rahul/PycharmProjects/rock_scissor_game/data/'
model = load_learner(train_directory,'export.pkl')


def imageToTensorImage(bgr_img):
    sz = 224
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    # crop to center to the correct size and convert from 0-255 range to 0-1 range
    H,W,C = rgb_img.shape
    rgb_img = rgb_img[(H-sz)//2:(sz +(H-sz)//2),(H-sz)//2:(sz +(H-sz)//2),:] / 256
    return vision.Image(px=pil2tensor(rgb_img, np.float32))


cap= cv2.VideoCapture(-1)
start = True
i=0
my_list = ['rock','paper','scissor']
choice = 0
pc_img = np.array([])
img_name=None
while True:
    ret, frame = cap.read()

    if ret!=True:
        break
    if start:
        x, y, w, h = 0, 100, 280, 300

        frame = cv2.rectangle(frame, (0, 100), (280, 400), (255, 255, 0), 4)
        img = frame[y:y + h, x:x + w]

        test_img= imageToTensorImage(img)
        font = cv2.FONT_HERSHEY_SIMPLEX

        frame=cv2.putText(frame,str(np.array(model.predict(test_img)[0])) , (100, 420), font, 1.0, (100, 255,255), 3, cv2.LINE_AA)
        frame=cv2.resize(frame,(1280,720))

        posx=1000
        posy=100

        for y in range(pc_img.shape[0]):
            if y+posy<frame.shape[0]:
                for x in range(pc_img.shape[1]):
                    if x+posx<frame.shape[1]:

                        frame[y+posy][x+posx]=pc_img[y][x]

        frame = cv2.putText(frame, img_name, (1000, 200), font, 1.0, (100, 255, 255), 3,
                            cv2.LINE_AA)
        cv2.imshow('full', frame)
        i+=1

    key =cv2.waitKey(30) & 0xff

    if key == ord('p'):
        start = True

    if key == ord('t'):
        start= True
        choice = np.random.randint(0, 3)

        pc_img = cv2.resize(cv2.imread('/home/rahul/PycharmProjects/rock_scissor_game/data/{}/frame{}.jpg'.format(my_list[choice], choice * 100)),
                            (280, 300))
        img_name = my_list[choice]

    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()

