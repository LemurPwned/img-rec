import cv2
import matplotlib.pyplot as plt
import pprint as pp
import os
from process_frame import find_lines
import numpy as np


def biplot(img):
    edges = cv2.Canny(img, 520, 120)
    [b, g, r] = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Standard frame")
    plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(edges)
    plt.title("Canny frame")
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close()
    cv2.destroyAllWindows()


def generate_dataset(path_to_read, path_to_write):
    """
    turns video into set of frames
    """
    capture = cv2.VideoCapture(path_to_read)
    count=0
    cv2.namedWindow("Standard Frame")
    while capture.isOpened():
        ret, frame = capture.read()
        try:
            cv2.imwrite(path_to_write+"\\"+str(count)+".jpg", frame)
            cv2.imshow('Standard Frame', frame)
            count+=1
            print(count)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    capture.release()
    cv2.destroyAllWindows()


def detect_blur(img, threshold):
    isBlurred = False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lp = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lp < threshold:
        isBlurred = True
    return isBlurred
    
    
class Vector:
    def __init__(self, frame, x, y):
        self.x = x
        self.y = y
        self.frame = frame
 
    def norm(self):
        return np.sqrt(self.x**2 + self.y**2)

    def dot(self, v1):
        return Vector(self.x*v1.x, self.y*v1.y)
    
    def calc_angle(self):
        self.angle = np.arcsin(float(self.x)/self.norm())
    
    def __str__(self):
        return ("(%u, %u) ; angle :(%3.2f) ; norm :(%3.2f) ; Frame:(%s) " %
                (self.x, self.y, self.angle, self.norm(), self.frame))
        
        
class ParamPack:
    def __init__(self, name, alphaL, betaL, alphaR, betaR):
        self.source = name
        self.alphaL = abs(alphaL)
        self.betaL = abs(betaL)
        self.alphaR = abs(alphaR)
        self.betaR = abs(betaR)
    
    def label(self, margin):
        """
        labels the data; margin controls sensitivity
        S - straight
        L - left turn
        R - right turn
        N stands for None
        """
        if self.alphaL == None or self.alphaR == None:
            self.label = "N"
        elif abs(self.alphaL - self.alphaR) <= margin:
            self.label = "S"
        elif (self.alphaL - self.alphaR) > margin:
            self.label = "L"
        elif -(self.alphaL - self.alphaR) > margin:
            self.label = "R"
        else:
            self.label = "N"
            
    def __str__(self):
        return ("L:(%u, %u) ; R:(%u, %u) ; Label:(%s) ; Frame:(%s) " %
                (self.alphaL, self.betaL, self.alphaR, self.betaR, self.label, self.source))


def write_to_txt(batch, filepath, typ='vector', verbose ='dataset'):
    """
    turns object dataset into .txt file
    """
    if typ == 'params':
        with open(filepath, "w") as txt:
            for param_pack in batch:
                txt.write("{} ; {} ; {} \n".format(param_pack.alphaL, 
                            param_pack.alphaR, param_pack.label))
                #txt.write(str(param_pack.alphaL) +", "+str(param_pack.alphaR) +"\n")
    elif typ == 'vector':
        if verbose == 'human':
            with open(filepath, "w") as txt:
                for vector in batch:
                    txt.write(str(vector[0]) + "\n")
                    txt.write(str(vector[1]) + "\n")
        elif verbose == 'dataset':
             with open(filepath, "w") as txt:
                for vector in batch:
                    #txt.write(str(vector[0].x) +";"+str(vector[0].y) +";"+ str(vector[0].angle) +";"+ str(vector[0].norm()) + "\n")
                    txt.write(str(vector[1].x) +";"+ str(vector[1].angle) +";"+ str(vector[1].norm()) + "\n")           

def vote(caseA, caseB, conflict):
    if caseA == caseB:
        return caseA
    else:
        if conflict == caseA:
            return caseA
        elif conflict == caseB:
            return caseB
        else:
            #Unresolved
            return 'N'
            
def eliminate_blur(directory):
    '''
    actually it was meant just to purify dataset, but it actually prepares whole dataset
    #####uncomment stuff to see the image with label, quality control xD########
    batch - batch of parameters of left and right side of the track
    return_set - see last line of function, set to return
    final_dataset - is the set of not blurred frames
    '''
    #cv2.namedWindow("Blur check")
    font = cv2.FONT_HERSHEY_SIMPLEX
    return_set = []
    final_dataset = []
    not_blurred = 0
    blurred = 0
    batch = []
    color = (0,0,0)
    dy = 10
    vector_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(directory + "\\" + filename)
            if img == None:
                break
            h,w,c = img.shape
            if detect_blur(img, 75):
                #cv2.putText(img, "blurred", (h/2 - (1/4)*h,w/2),
                #font, 1, (0,0,255))
                blurred += 1
            else:                
                '''
                cv2.putText(img, "not blurred", (h/2 - (1/4)*h,w/2),
                            font, 1, (0,0,255))
                '''
                not_blurred += 1
                final_dataset.append(filename)
                lines_detected = find_lines(cv2.Canny(img, 520, 160), 10)
                for i in range(0, len(lines_detected)):
                    lines_coords = lines_detected[i]  # If one of the line is invalid, do not draw it                
                               
                    '''
                    THIS CAUSES THE BLOODY FUCKUP. WHAT THE FUCK???
                    if lines_coords[0].left_x == -1 or lines_coords[0].right_x == -1 or lines_coords[
                        1].left_x == -1 or lines_coords[1].right_x == -1:
                        continue
                    '''
                    
                    if (not lines_coords[0].is_valid) or (not lines_coords[1].is_valid()):
                        continue 
                        
                        
                    #below testing vector approach
                    vectorLeft = Vector(filename, lines_coords[0].right_x - lines_coords[0].left_x, dy)
                    vectorRight = Vector(filename, lines_coords[1].right_x - lines_coords[1].left_x, dy)
                    vectorLeft.calc_angle()
                    vectorRight.calc_angle()
                    vector_list.append([vectorLeft, vectorRight])
                    
                    
                    params = ParamPack(filename, float(lines_coords[0].right_x - lines_coords[0].left_x),
                                           float(lines_coords[0].right_y - lines_coords[0].left_y),
                                           float(lines_coords[1].right_x - lines_coords[1].left_x),
                                           float(lines_coords[1].right_y - lines_coords[1].left_y))
                    params.label(1)
                    batch.append(params)
                    '''  
                    #below some tests to find out why some frames don't display labels
                    if params.label == 'L':
                        #red for turn left
                        color = (0,0,255)
                    elif params.label == 'R':
                        #green for turn right
                        color = (0,255,0)
                    else:
                        #blue for straight
                        color = (255,0,0)
                    #some invalid frames, always "valid" though
                    if filename in ['114.jpg','102.jpg', '118.jpg', '120.jpg', '121.jpg', '122.jpg']:
                        print(params)
                        if lines_coords[1].is_valid():
                            print("Valid")
                        else:
                            print("Invalid")
                            
                        
                    cv2.putText(img, str(params.label), (lines_coords[0].right_x,lines_coords[0].right_y),
                                font, 1, color)
                    cv2.putText(img, str(params.label), (lines_coords[1].left_x,lines_coords[1].left_y),
                                font, 1, color)
                                   
                print("Displaying : {}".format(filename))                    
                cv2.imshow("Blur check", img)
                key = cv2.waitKey(0)
                if key == 13:
                    break
                cv2.destroyAllWindows()
                '''
            print("Analyzed the file : " + str(filename))

        else:
            break
    return_set = [float(not_blurred) / (blurred + not_blurred), final_dataset, batch, vector_list]
    return return_set

if __name__ == "__main__":
	#biplot(img)
	#generate_dataset("D:\\Dokumenty\\image-recognition\\track\\track.avi", "D:\\Dokumenty\\image-recognition\\dataset")
	to_return = eliminate_blur("D:\\Dokumenty\\image-recognition\\dataset")
	#print((to_return[2][2]))
	#for param_set in to_return[2]:
	#    param_set.label(2)
	#write_to_txt(to_return[2], "D:\\Dokumenty\\image-recognition\\parameters\\params.txt", 'params' ,'human')
	#write_to_txt(to_return[3], "D:\\Dokumenty\\image-recognition\\parameters\\vectorsL.txt", 'vector', 'dataset')
	#write_to_txt(to_return[3], "D:\\Dokumenty\\image-recognition\\parameters\\vectorsR.txt", 'vector', 'dataset')



