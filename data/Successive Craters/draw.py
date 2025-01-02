import cv2
def parse_params(input_str):
    # Remove brackets and split by comma, then convert to float
    input_str = input_str.strip()[1:-1]  # Remove the brackets
    return list(map(float, input_str.split(',')))

def draw_ellipse_on_image(image_number):
    
    base_path = "/Users/vasilismichalakis/Documents/Uni/2nd Year/Sem 2/Topics/Code.nosync/christian_cid_method-main/data/CH5-png"
    path = base_path + f'/{image_number}.png'

    # Load the image
    img = cv2.imread(path)

    if img is None:
        print("Error: Could not load image.")
        return

    height = 1728
    width = 2352

    # Ask the user for ellipse parameters
    params_input = input("Enter ellipse parameters as a list: ")
    params = parse_params(params_input)
    
    x_center = params[0] * width/100
    y_center = params[1] * height/100

    axis_length1 = params[2] * width/100 * 2
    axis_length2 = params[3] * height/100 * 2

    angle = params[4]

    color = (0, 0, 255) 

    thickness = 2

    # Draw the ellipse
    cv2.ellipse(img, ((x_center,y_center), (axis_length1,axis_length2), angle), color, thickness)

    # Show the image with the ellipse
    cv2.imshow(image_number, img)

if __name__ == "__main__":
    while True:

        image_number = input("Enter image number or 'q' to quit: ")

        # Exit condition if 'q' is entered
        if image_number.lower() == 'q':
            break

        # Call the function to draw ellipse on the image
        draw_ellipse_on_image(image_number)

        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
