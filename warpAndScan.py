import cv2
import numpy as np
import utlis
import easyocr
import re

image="aryansh6.jpg"

def warpAndScan(image):

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # cv2 is for computer vision
    # imshow()
    # imread()
    # resize, blur etc etc hote hai
    #  
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image}. Please check the file path or its integrity.")

    print("file read")

    h=1600
    w=1200
    
    heightImg=800
    widthImg=600

    original_image=img

    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    original_image=cv2.resize(original_image, (w, h))
    # ocr needs high res image taaki ocr sahi se padh paaye

    # jitna bhi image processing tha
    # like finding rectangles, biggest rectangles 
    # uske liye you dont need high res
    # but for scanning text we need high res 
    # toh jitna processing wala part hai woh hum 800 by 600 ki image karte hai


    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)  # ADD GAUSSIAN BLUR
    thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    
    ## FIND ALL CONTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    # print(biggest.size)
    text = ""  # Default value for text
    if biggest.size != 0:
        # print("entered if block")
        biggest_original=np.copy(biggest)
        # print(biggest_original[0][0][1])
        biggest_original[0][0][0]*=int(w/widthImg)
        biggest_original[0][0][1]*=int(w/widthImg)

        biggest_original[1][0][0]*=int(w/widthImg)
        biggest_original[1][0][1]*=int(w/widthImg)

        biggest_original[2][0][0]*=int(w/widthImg)
        biggest_original[2][0][1]*=int(w/widthImg)

        biggest_original[3][0][0]*=int(w/widthImg)
        biggest_original[3][0][1]*=int(w/widthImg)
        # print("biggest",biggest)
        # print(biggest.shape)

        # print("biggest original", biggest_original)
        
        biggest = utlis.reorder(biggest)
        # 4 points of the biggest rectangle 800 by 600 ki image
        biggest_original=utlis.reorder(biggest_original)
        # 4 points of the biggest rectangle 1600 by 1200 ki image

        
        
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts1_original=np.float32(biggest_original)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        pts2_original=np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        matrix_original=cv2.getPerspectiveTransform(pts1_original, pts2_original)
        
        
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg),flags=cv2.INTER_CUBIC)
        imgWarpColored_original=cv2.warpPerspective(original_image, matrix_original, (w, h),flags=cv2.INTER_CUBIC)
        # # REMOVE 20 PIXELS FROM EACH SIDE
        # imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        # imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
        


        imgWarpColored_original=cv2.resize(imgWarpColored_original,(1300,840))


        

        # Assuming the necessary variables and imports are already defined above.
        print("creating blank image ")
        # Save the blank image for testing/debugging
        cv2.imwrite("./balihaar/filteredImages/0imgBlank.jpg", imgBlank)

        # Save the grayscale image
        print("creating grayscale image ")
        
        cv2.imwrite("./balihaar/filteredImages/1imgGray.jpg", imgGray)

        # Save the blurred image
        print("creating blurred image ")

        cv2.imwrite("./balihaar/filteredImages/2imgBlur.jpg", imgBlur)

        # Save the thresholded image
        print("creating threshold image ")

        cv2.imwrite("./balihaar/filteredImages/3imgThreshold.jpg", imgThreshold)

        # Save the image with all contours drawn
        print("creating contours image ")

        cv2.imwrite("./balihaar/filteredImages/4imgContours.jpg", imgContours)

        # Save the image with the biggest contour highlighted
        print("creating biggest contour image ")

        cv2.imwrite("./balihaar/filteredImages/5imgBigContour.jpg", imgBigContour)

        if biggest.size != 0:
            # Save the warped colored image
            cv2.imwrite("./balihaar/filteredImages/6imgWarpColored.jpg", imgWarpColored)

            # Save the original warped colored image resized to (1300, 840)
            cv2.imwrite("./balihaar/filteredImages/7imgWarpColored_original.jpg", imgWarpColored_original)


        print("reader initialized")
        reader = easyocr.Reader(['en','hi'])
        # text recognition model easyOCR
        print("reading")
        result = reader.readtext(imgWarpColored_original)
        print("file read")

        text = ' '.join([entry[1] for entry in result])
        print("text:",text)

    return text

     



def frontOrBack(text):
    if "Address" in text or "A dd ress" in text or "Addre ss" in text or "Addrèss" in text or "Addres55" in text or "4ddress" in text or "Addrass" in text or "Add ress" in text or "Addre$$" in text or "sserdA" in text or "Addre5s" in text:
        return "back"
    else:
        return "front"

def extractFront(text):
        
        # searching for name in front 
        start = text.find("DO8")
        if(start==-1):
            start = text.find("D08")
        if(start==-1):
            start = text.find("D08")
        if(start==-1):
            start = text.find("D0B")
        if(start==-1):
            start = text.find("DOB")



        print(start)
        name=""
        english="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        firstinstace=-1
        secondInstanceace=-1
        for i in range(start-1,-1,-1):

            if text[i] in english:
                secondInstance=i
                break
        for i in range(i-1,-1,-1):

            if text[i] not in english:
                if(text[i].isspace()==False):
                    firstinstace=i+1
                    break
            

        name=text[firstinstace+1:secondInstance+1]
        # for dob

        dob_pattern = r"(\d{2}/\d{2}/\d{4})"
        dob_match = re.search(dob_pattern, text)
        dob = dob_match.group(1) if dob_match else None
        # dob searched

        # for id number
        aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
        aadhar_match = re.search(aadhar_pattern, text)

        if aadhar_match:
            id= aadhar_match.group() if aadhar_match else None
        else:
           pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"
           pan_match= re.search(pan_pattern, text)
           id = pan_match.group() if pan_match else None

        return name, dob, id


def extractBack(extracted_text):
    # Extract text between "address" and a 12-digit number
    
    substring = "Address:"
    start = extracted_text.find(substring)
    if start is None:
        return "null"
    if(start==-1):
        start = extracted_text.find("Address")
    if(start==-1):
        start = extracted_text.find("address")
    if(start==-1):
        start = extracted_text.find("address:")
    
    start_index=start+len("Address:")
    

    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    aadhar_match = re.search(aadhar_pattern, extracted_text)
    end_index = aadhar_match.start()

    return extracted_text[start_index:end_index]
    



    
text=warpAndScan(image)
id, address, dob, name= None, None, None, None
frontOrBack = frontOrBack(text)
if frontOrBack == "back":
    address = extractBack(text)
    
if frontOrBack == "front":
    id, dob, name = extractFront(text)

print(id, dob, name, address)
