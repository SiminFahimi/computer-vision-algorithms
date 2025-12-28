import numpy as np
class Kernel:
    def __init__(self,shape=None):

        self.shape=shape
        
        if self.shape==None:
            self.center=(0,0)
        else:
            self.center = (shape[0]//2 , shape[1]//2)

        self.kernel=np.ones(shape)

        self.height = shape[0]   
        self.width  = shape[1]   

        self.coordinate=self.coordinates()
    
    def coordinates(self):
        x = np.arange(self.width) - self.center[1]   
        y = np.arange(self.height)  - self.center[0]   
        return np.meshgrid(x, y)
    

class Gaussian_kernel(Kernel):

    def __init__(self,sigma,shape):

        super().__init__(shape)
        self.sigma=sigma
        self.return_kernel()

    def return_kernel(self):

        X,Y=self.coordinate
        kernel=1 / (2 * np.pi * self.sigma**2)* np.exp(-((X**2)+(Y**2))/(2*(self.sigma**2)))
        kernel = kernel / kernel.sum()
        self.kernel=kernel
        return 

class Custom_kernel(Kernel):
    def __init__(self,shape):
        super().__init__(shape)
        self.return_kernel()


    def return_kernel(self):
        center=self.center
        self.kernel[center[0],center[1]]=-1
        return
    
def add_filter(image,mask: Kernel):
    center_y,center_x=mask.center
    image_height,image_width=image.shape
    mask_height=mask.height
    mask_width=mask.width
    
    padded = np.pad(image, ((center_y,mask_height-1-center_y), (center_x,mask_width-center_x-1)),
                     'constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded[i : i + mask.height, j : j + mask.width]
            filtered_image[i, j] = np.sum(mask.kernel * region)
    return filtered_image 
