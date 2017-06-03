import cv2 

def resize(img, width, height)
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

