import cv2 
import dlib

def resize(img, width, height):
    new_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return new_img

def process_edge_image(input, output):
    print('edge', input, output)
    img = cv2.imread(input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    edges = cv2.Canny(img, ret * 0.5, ret)

    cv2.imwrite(output, 255 - edges)
    return 255 - edges

def get_face_position(img_src):
    img = cv2.imread(img_src)
    h, w, _ = img.shape
    
    detector = dlib.get_frontal_face_detector()

    dets = detector(img, 1)

    for i, face in enumerate(dets):
        width = face.right() - face.left()
        height = face.bottom() - face.top()

        left = face.left() - width // 2 
        right = face.right() + width // 2 
        top = face.top() - height // 2
        bottom = face.bottom() + height // 2
        
        if left < 0: left = 0
        if right > w: right = w
        if top < 0: top = 0 
        if bottom > h: bottom = h 


        return True, {
            'top': top,
            'bottom': bottom,
            'right': right,
            'left': left,
        }

    return False, {
        'top': None,
        'bottom': None,
        'right': None,
        'left': None,
    }
