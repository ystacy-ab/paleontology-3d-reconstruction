from rembg import remove, new_session
import cv2

session = new_session("isnet-general-use")

input_img = cv2.imread('images/image3.jpg')
output_rgba = remove(input_img, session=session)

alpha_channel = output_rgba[:, :, 3]

_, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

cv2.imwrite('masks_code/image3_rembg_mask.png', mask)

# cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()