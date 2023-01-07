import cv2  # used for capturing and displaying the video stream
import torch  # used for loading and running the model
import matplotlib.pyplot as plt  # used for displaying the monocular depth prediction
import timm  # contains utility functions for PyTorch

# Load the MiDaS model using the torch.hub API
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')

# Move the model to the CPU and set it to evaluation mode
midas.to('cpu')
midas.eval()

# Load the transforms for preprocessing the input images using the torch.hub API
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Create a video capture object for the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Start a loop to process the video frame-by-frame
while cap.isOpened():  # continue as long as the video capture object is open
    ret, frame = cap.read()  # read a frame of the video

    if ret:  # if the frame was successfully read
        # Convert the frame from the BGR color space to the RGB color space
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the transform to the frame
        imgbatch = transform(img).to('cpu')

        # Use the model to get a prediction of the monocular depth
        with torch.no_grad():
            prediction = midas(imgbatch)

            # Post-process the prediction to resize it to the size of the original frame
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = img.shape[:2],
                mode = 'bicubic',
                align_corners=False
            ).squeeze()

            # Convert the prediction to a numpy array
            output = prediction.cpu().numpy()

        # Display the monocular depth prediction
        plt.imshow(output)

        # Display the original frame
        cv2.imshow('CV2Frame',frame)

        # Pause the loop for a short time
        plt.pause(0.00001)

        # Check if the user has pressed the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # If the 'q' key has been pressed, release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()

# Ensure that the plots are displayed
plt.show()
