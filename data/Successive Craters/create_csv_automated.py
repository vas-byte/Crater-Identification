import json
import csv
import os
import cv2

# Read params from file
def read_json(i):
    cwd = os.getcwd()
    path = cwd + f'/Params/{i}.png.json'
    jsonFile = open(path, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file
    return data

# Write remaining unused params back to JSON file
def write_json(i, data):
    cwd = os.getcwd()
    path = cwd + f'/Params/{i}.png.json'
    jsonFile = open(path, "w")
    jsonFile.write(json.dumps(data))
    jsonFile.close()

# Sort list of candidate params
def get_ordered_list(points, x, y):
   points.sort(key = lambda p: (p[0] - x)**2 + (p[1] - y)**2)
   return points

# Draw ellipse params
def draw_ellipse_on_image(image_number, param):
    
    base_path = "/Users/vasilismichalakis/Documents/Uni/2nd Year/Sem 2/Topics/Code.nosync/christian_cid_method-main/data/CH5-png"
    path = base_path + f'/{image_number}.png'

    # Load the image
    img = cv2.imread(path)

    if img is None:
        print("Error: Could not load image.")
        return

    height = 1728
    width = 2352
 
    x_center = param[0] * width/100
    y_center = param[1] * height/100

    axis_length1 = param[2] * width/100 * 2
    axis_length2 = param[3] * height/100 * 2

    angle = param[4]

    color = (0, 0, 255) 

    thickness = 2

    # Draw the ellipse
    cv2.ellipse(img, ((x_center,y_center), (axis_length1,axis_length2), angle), color, thickness)

    # Show the image with the ellipse
    cv2.imshow(str(image_number), img)

if __name__ == "__main__":

    # List of params to write
    params = []

    # Successive Crater ID
    successive_crater_id = 1336
    start_frame = 410

    # Read from 0.png.json (to get ellipse parameters)
    first = read_json(start_frame)
    param = first[0]
    first.remove(param)
    write_json(start_frame,first)
    last_img_num = start_frame
    params.append([param,start_frame])

    # Num frames
    num_frames = 0

    should_stop = False

    if start_frame < 100:
        for i in range(start_frame+1,101):
            potential_params = read_json(i)
            isolated_params = []
            should_stop = False

            if len(potential_params) == 0:
                num_frames += 1.5
                params.append(["NULL",i])
                print(f"Frame {i} does not have successive ellipse")
                continue

            if param[0] > 50:
            
                for p in potential_params:
                    if p[0] - param[0] < max(4,num_frames) and p[0] - param[0] > -0.75 and p[1] > param[1] - 1.75:
                        isolated_params.append(p)
                
                isolated_params = get_ordered_list(isolated_params,param[0],param[1])

            else:
                
                for p in potential_params:
                    if p[0] - param[0] > min(-4,-num_frames) and p[0] - param[0] < 0.75 and p[1] > param[1] - 1.75:
                        isolated_params.append(p)

                isolated_params = get_ordered_list(isolated_params,param[0],param[1])
            
            if len(isolated_params) == 0:
                num_frames += 1.5
                params.append(["NULL",i])
                print(f"Frame {i} does not have successive ellipse")
            
            for p in isolated_params:
                draw_ellipse_on_image(last_img_num, param)
                draw_ellipse_on_image(i,p)
                cv2.moveWindow(str(i),500,-20)
                cv2.waitKey(1)
                print("Y for yes, N for No, None for no match, -1 to stop")
                user_input = input("Do these match (Y, N, None, -1): ")

                if user_input == "-1":
                    cv2.destroyAllWindows()
                    should_stop = True
                    break
                elif user_input == "None":
                    num_frames += 1.5
                    params.append(["NULL",i])
                    cv2.destroyAllWindows()
                    break
                elif user_input == "N":
                    cv2.destroyAllWindows()

                    if len(isolated_params) == 1 or p == isolated_params[-1]:
                        num_frames += 1.5
                        params.append(["NULL",i])
                        print(f"Frame {i} does not have successive ellipse")

                    continue
                elif user_input == "Y":
                    cv2.destroyAllWindows()
                    print(p)
                    params.append([p,i])
                    potential_params.remove(p)
                    write_json(i,potential_params)
                    param = p
                    last_img_num = i
                    num_frames = 0
                    break
                    
            if should_stop:
                break
    
    next_ten = 110

    if start_frame >= 110:
        next_ten = start_frame + 10
        
    for i in range(next_ten,420,10):

        if should_stop:
            break

        potential_params = read_json(i)
        isolated_params = []
        should_stop = False

        if len(potential_params) == 0:
            params.append(["NULL",i])
            print(f"Frame {i} does not have successive ellipse")
            continue
        
        if i < 270:
            if param[0] > 50:
            
                for p in potential_params:
                    if p[0] - param[0] < 20 and p[0] - param[0] > -5 and p[1] > param[1] - 12:
                        isolated_params.append(p)
                
                isolated_params = get_ordered_list(isolated_params,param[0],param[1])

            else:
                
                for p in potential_params:
                    if p[0] - param[0] > -20 and p[0] - param[0] < 5 and p[1] > param[1] - 12:
                        isolated_params.append(p)

                isolated_params = get_ordered_list(isolated_params,param[0],param[1])
        
        elif start_frame < 270 and last_img_num < 270:
            
            for p in potential_params:
                    if p[1] < 30:
                        isolated_params.append(p)

            isolated_params = get_ordered_list(isolated_params,param[0],param[1])
             
        else:

            for p in potential_params:
                    if abs(p[0] - param[0]) < 25 and abs(p[1] - param[1]) < 65:
                        isolated_params.append(p)
                
            isolated_params = get_ordered_list(isolated_params,param[0],param[1])
        
        if len(isolated_params) == 0:
            params.append(["NULL",i])
            print(f"Frame {i} does not have successive ellipse")
        
        for p in isolated_params:
            draw_ellipse_on_image(last_img_num, param)
            draw_ellipse_on_image(i,p)
            cv2.waitKey(1)
            print("Y for yes, N for No, None for no match, -1 to stop")
            user_input = input("Do these match (Y, N, None, -1): ")

            if user_input == "-1":
                cv2.destroyAllWindows()
                should_stop = True
                break
            elif user_input == "None":
                params.append(["NULL",i])
                cv2.destroyAllWindows()
                break
            elif user_input == "N":
                cv2.destroyAllWindows()

                if len(isolated_params) == 1 or p == isolated_params[-1]:
                    params.append(["NULL",i])
                    print(f"Frame {i} does not have successive ellipse")

                continue
            elif user_input == "Y":
                cv2.destroyAllWindows()
                print(p)
                params.append([p,i])
                potential_params.remove(p)
                write_json(i,potential_params)
                param = p
                last_img_num = i
                break

    if os.path.isfile(f'crater_{successive_crater_id}.csv'):
        print("Error! File Exists!")

        with open(f'temp.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Parameters","Group ID","Image"])

            for p in params:
                writer.writerow([p[0],successive_crater_id,p[1]])

    else:
        with open(f'crater_{successive_crater_id}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Parameters","Group ID","Image"])

            for p in params:
                writer.writerow([p[0],successive_crater_id,p[1]])
               
    
