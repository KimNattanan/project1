
import numpy as np
import cv2
import copy
import random

from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image

stack=""
items=open("items.txt","r")
for e in items: stack=stack+e
stack=stack.split('\n')
stack0=copy.deepcopy(stack)
random.shuffle(stack)
result=open("result.txt","w")

def detect_face(_image,draw=0):
    image = _image
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces = detect_faces(image)
    if(not draw): return faces
    if not len(faces):
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
        render_to_image(render_data, image)
        return image
    
def rotate(points, ANGLE,deg=0):
    if(deg): ANGLE = np.deg2rad(ANGLE)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    ).astype(int)

vid=cv2.VideoCapture(0)
if not vid.isOpened():
    print("Cannot open camera")
    exit()


class INFO:
    def __init__(self,sz,col):
        self.sz=sz
        self.col=col

class BOX:
    def __init__(self,info):
        self.info=copy.deepcopy(info)
        self.t0=0
        self.t1=0
        self.state=0
        self.val="??"

time=0;
info0=INFO((150,40),(255,255,255));
info1=INFO((180,70),(0,255,0));
info2=INFO((180,70),(255,0,255));
info3=INFO((150,40),(150,150,150))
boxL=BOX(info0);
boxR=BOX(info0);
offX=77
offY=100
low=30
select_t=1
class GAME:
    def __init__(self):
        self.state=0
        self.randL=1
        self.randR=1
        self.rand_t=7
        self.rand_t2=3
        self.t0=0
        self.t1=0
    def roll(self,box):
        box.info=copy.deepcopy(info3)
        box.val=copy.deepcopy(stack0[random.randint(0,len(stack0)-1)])
    def roll2(self,box):
        if(not len(stack)):
            box.val="??"
            return
        box.val=copy.deepcopy(stack[-1])
        stack.pop()
    def upd(self):
        if(self.state==2): return
        global boxL
        global boxR
        if(self.state==0):
            self.t0=copy.deepcopy(time)
            self.t1=time+self.rand_t*10
            self.state=1
        if(time<self.t1):
            if(self.randL): self.roll(boxL)
            if(self.randR): self.roll(boxR)
        else:
            self.state=2
            if(self.randL):
                self.randL=0
                self.roll2(boxL)
            if(self.randR):
                self.randR=0
                self.roll2(boxR)
            boxL.info=copy.deepcopy(info0)
            boxR.info=copy.deepcopy(info0)
            if(self.rand_t!=self.rand_t2): self.rand_t=self.rand_t2
game=GAME()

def upd(frame,faces):
    frame2=copy.deepcopy(frame)
    x0,y1=faces[0].data[0]
    x1,y0=faces[0].data[1]
    leye=faces[0].data[2]
    reye=faces[0].data[3]
    dy=(leye[1]-reye[1])*frame.shape[0]
    x0=int(x0*frame.shape[1])
    x1=int(x1*frame.shape[1])
    xmid=int((x0+x1)/2)
    y0=int((y0)*frame.shape[0])
    y1=int((y1)*frame.shape[0])

    if(game.state==2):
        if(abs(dy)<=low):
            boxL.info=copy.deepcopy(info0)
            boxR.info=copy.deepcopy(info0)
            boxL.state=0
            boxR.state=0
        elif(dy>0):
            boxR.info=copy.deepcopy(info0)
            boxL.info.sz=copy.deepcopy(info1.sz)
            boxR.state=0
            if(boxL.state==0):
                boxL.state=1
                boxL.t0=copy.deepcopy(time)
                boxL.t1=time+select_t*10
            elif(boxL.state==1 and time<boxL.t1):
                r=(time-boxL.t0)/(boxL.t1-boxL.t0)
                dcol=(info1.col[0]-info0.col[0],info1.col[1]-info0.col[1],info1.col[2]-info0.col[2])
                col_now=(info0.col[0]+r*dcol[0],info0.col[1]+r*dcol[1],info0.col[2]+r*dcol[2])
                col_now=(int(col_now[0]),int(col_now[1]),int(col_now[2]))
                boxL.info.col=copy.deepcopy(col_now)
            elif(boxL.state==1 and time>=boxL.t1):
                boxL.state=2
                boxL.info=copy.deepcopy(info2)
                game.state=0
                game.randL=0
                game.randR=1
                result.write(boxL.val+" > "+boxR.val+"\n")

        else:
            boxL.info=copy.deepcopy(info0)
            boxR.info.sz=copy.deepcopy(info1.sz)
            boxL.state=0
            if(boxR.state==0):
                boxR.state=1
                boxR.t0=copy.deepcopy(time)
                boxR.t1=time+select_t*10
            elif(boxR.state==1 and time<boxR.t1):
                r=(time-boxR.t0)/(boxR.t1-boxR.t0)
                dcol=(info1.col[0]-info0.col[0],info1.col[1]-info0.col[1],info1.col[2]-info0.col[2])
                col_now=(info0.col[0]+r*dcol[0],info0.col[1]+r*dcol[1],info0.col[2]+r*dcol[2])
                col_now=(int(col_now[0]),int(col_now[1]),int(col_now[2]))
                boxR.info.col=copy.deepcopy(col_now)
            elif(boxR.state==1 and time>=boxR.t1):
                boxR.state=2
                boxR.info=copy.deepcopy(info2)
                game.state=0
                game.randL=1
                game.randR=0
                result.write(boxR.val+" > "+boxL.val+"\n")
    else:
        game.upd()

    obj0=np.array([(xmid-offX-boxL.info.sz[0],y1-offY),
                   (xmid-offX,y1-offY),
                   (xmid-offX,y1-offY-boxL.info.sz[1]),
                   (xmid-offX-boxL.info.sz[0],y1-offY-boxL.info.sz[1])])
    obj1=np.array([(xmid+offX,y1-offY),
                   (xmid+offX+boxR.info.sz[0],y1-offY),
                   (xmid+offX+boxR.info.sz[0],y1-offY-boxR.info.sz[1]),
                   (xmid+offX,y1-offY-boxR.info.sz[1])])
    cv2.drawContours(frame2,[obj0],0,boxL.info.col,-1,cv2.LINE_AA)
    cv2.drawContours(frame2,[obj1],0,boxR.info.col,-1,cv2.LINE_AA)
    cv2.putText(frame2,boxL.val,obj0[0],cv2.FONT_ITALIC,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame2,boxR.val,obj1[0],cv2.FONT_ITALIC,1,(0,0,0),2,cv2.LINE_AA)
    return frame2


while True:
    ret, frame = vid.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame=cv2.flip(frame,1)
    frame2=frame
    faces=detect_face(Image.fromarray(frame))
    time+=1
    if(time>=16777214): time=0

    if(len(faces)): frame2=upd(frame,faces)

    print(boxL.val,boxR.val);
    cv2.imshow('frame', frame2)
    key=cv2.waitKey(1)
    if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) <1:
        break

vid.release()
cv2.destroyAllWindows()
