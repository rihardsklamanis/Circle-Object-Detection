import sys
import cv2 as cv
import numpy as np

def main(argv):
    # Image file to search for circles
    filename = 'testImage.png'

    # Load the image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Check if the image is loaded successfully
    if src is None:
        print('Error loading the image!')
        return -1

    # Convert the image to grayscale and apply median blur
    gray = cv.medianBlur(cv.cvtColor(src, cv.COLOR_BGR2GRAY), 5)

    # Apply Hough Circle Transform to detect circles
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0] // 8,
                              param1=100, param2=30, minRadius=1, maxRadius=30)

    # Check if circles are detected
    if circles is not None:
        # Convert the coordinates and radii to integers
        circles = np.uint16(np.around(circles))

        # Display the image with filled circles, centers, and coordinates
        for center_x, center_y, radius in circles[0, :]:
            # Fill the circle with green color
            cv.circle(src, (center_x, center_y), radius, (0, 255, 0), thickness=-1)

            # Draw a small circle at the center of the detected circle
            cv.circle(src, (center_x, center_y), 3, (0, 0, 255), thickness=-1)

            # Calculate the starting point for text to be above the circle
            text_x = int(center_x - 0.5 * len(f"({center_x}, {center_y})"))
            text_y = int(center_y - radius - 5)

            # Display the center coordinates as text (in black color)
            cv.putText(src, f"({center_x}, {center_y})", (text_x, text_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Display the image with filled circles, centers, and coordinates
        cv.imshow("Result", src)

        # Save the new image with filled circles, centers, and coordinates
        output_filename = "testImageResult.png"
        cv.imwrite(output_filename, src)

        # Wait for a key press and close the window
        cv.waitKey(0)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
