simple line segmentation

For red line segmentation,
origin image -> extract saturation component -> opening -> threshold -> dilate

1. origin image
![Alt text](/simpleVersion/result/origin_figure.png?raw=true "origin image")

2. extract saturation compenent
![Alt text]('simpleVersion/result/saturation_figure.png?raw=true "saturation component")

3. opening
![Alt text]('/simpleVersion/result/img_open.png?raw=true "opening process")

4. threshold
![Alt text]('/simpleVersion/result/cv2.THRESH_BINARY + cv2.THRESH_OTSU + blur.png?raw=true "threshold result")



5. dilate
