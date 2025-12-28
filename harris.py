import kernel_create as krn
import numpy as np

def harris_corner_detection(image):
    g_mask=krn.Gaussian_kernel(2,(13,13))
    horizental_mask=krn.Custom_kernel((1,2))
    vertical_mask=krn.Custom_kernel((2,1))

    x_gradiant=krn.add_filter(image,horizental_mask)
    y_gradiant=krn.add_filter(image,vertical_mask)
    
    Ix2=krn.add_filter(np.pow(x_gradiant,2),g_mask)
    Iy2=krn.add_filter(np.pow(y_gradiant,2),g_mask)
    IxIy=krn.add_filter(np.multiply(x_gradiant,y_gradiant),g_mask)

    # tensor_structure=np.array([[Ix2,IxIy],[IxIy,Iy2]])
    # R=np.linalg.det(tensor_structure) - 0.4*((np.linalg.trace(tensor_structure))**2)
    k=0.04
    R = (Ix2*Iy2 - IxIy**2) - k*(Ix2 + Iy2)**2

    threshold = 0.01 * np.max(R)

    corners = np.zeros_like(R)

    for i in range(1, R.shape[0]-1):
        for j in range(1, R.shape[1]-1):

            window = R[i-1:i+2, j-1:j+2]

            if R[i, j] == np.max(window) and R[i, j] > threshold:
                corners[i, j] = 1
    return corners

    