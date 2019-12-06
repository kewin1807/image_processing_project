from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import click


def create_template(template):
    ref = cv2.imread(template, 0)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    digits = {}

    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi
    return digits


@click.command()
@click.option("--img", "-i", required=True,
              help="path to input image")
@click.option("--template", "-t", required=True, help="path to template")
def main(img, template):

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel

    digits = create_template(template)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a tophat (whitehat) morphological operator to find light
    # tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # # compute the Scharr gradient of the tophat image, then scale
    # gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
    #                   ksize=-1)
    # gradX = np.absolute(gradX)
    # (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    # gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    # gradX = gradX.astype("uint8")

    # # apply threshold otsu + closing operation to the binary image
    # gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    # thresh = cv2.threshold(gradX, 0, 255,
    #                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # # apply a second closing operation to the binary image, again
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)

    # cnts = imutils.grab_contours(cnts)
    # locs = []
    # # filter bounding box contains number
    # for (i, c) in enumerate(cnts):
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     ar = w / float(h)

    #     if ar > 2.5 and ar < 4.0:
    #         if (w > 40 and w < 55) and (h > 10 and h < 20):
    #             locs.append((x, y, w, h))
    # locs = sorted(locs, key=lambda x: x[0])
    # output = []
    # for (i, (gX, gY, gW, gH)) in enumerate(locs):

    #     groupOutput = []
    #     group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    #     group = cv2.threshold(group, 0, 255,
    #                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #     digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
    #                                  cv2.CHAIN_APPROX_SIMPLE)
    #     digitCnts = imutils.grab_contours(digitCnts)
    #     digitCnts = contours.sort_contours(digitCnts,
    #                                        method="left-to-right")[0]
    #     # text = pytesseract.image_to_string(group, config=config)
    #     # print(text)

    #     # matching template with each digit
    #     for c in digitCnts:
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         roi = group[y:y + h, x:x + w]
    #         roi = cv2.resize(roi, (57, 88))

    #         # initialize a list of template matching scores
    #         scores = []
    #         for (digit, digitROI) in digits.items():
    #             # apply correlation-based template matching, take the
    #             # score, and update the scores list
    #             result = cv2.matchTemplate(roi, digitROI,
    #                                        cv2.TM_CCOEFF)
    #             (_, score, _, _) = cv2.minMaxLoc(result)
    #             scores.append(score)

    #     # the classification for the digit ROI will be the reference
    #     # digit name with the * largest * template matching score
    #         groupOutput.append(str(np.argmax(scores)))

    #     cv2.rectangle(image, (gX - 5, gY - 5),
    #                   (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    #     cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # # print("Credit Card #: {}".format("".join(output)))

    cv2.imshow("Image", gray)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
