import numpy as np
import skimage as skim


def sphere_idx(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    assert len(position) == len(shape)
    n = len(shape)
    position = np.array(position).reshape((-1,) + (1,) * n)
    arr = np.linalg.norm(np.indices(shape) - position, axis=0)
    return arr <= radius



def idealised_Nd_image(shape = [500,500,500], radius = 50, number = 10):
    '''
    Generate an n-dimensional (size and dimensions defined by _shape_) "idealised image" for segmentation, consisting of _number_ ND spheres of radius _radius_ in white (1) on a dark background (0)
    '''
    import numpy as np

    blank_canvas = np.zeros(shape)
    coords = np.random.randint(0,shape[0],size=(number,len(shape)))

    for i in range(coords.shape[0]):
        blank_canvas = blank_canvas + (np.ones(shape) * sphere_idx(shape,radius,coords[i,:]))

    return (blank_canvas>0).astype(float)



def create_idealised_image(shape=(1000,1000),radius = 25, number  = 50, random_stream = None, random_shapes = False):
    '''
    Generate a 2D "idealised image" for segmentation, consisting of ND spheres in white (1) on a dark background (0)

    shape: the x and y dimensions of the image to be created
    radius: radius of cicrcles be generated (if not using random_shapes)
    number: number of shapes to be created on the canvas
    random_stream: random state to be used for coordinates of shapes
    random_shapes: Boolean flag, to create scikitimage's draw.random_shapes module instead of circles
    '''
    import numpy as np

    if random_stream == None:
        random_stream = np.random.RandomState()
        random_stream.seed(0)

    blank_canvas = np.zeros(shape)
    coords = np.random.randint(0,shape[0],size=(number,2))
    coords = random_stream.randint(0,shape[0],size=(number,2))

    for i in range(coords.shape[0]):
        rr, cc = skim.draw.disk((coords[i,0], coords[i,1]), radius,shape=shape)
        blank_canvas[rr, cc] = 1
        # rr, cc = skim.draw.disk((coords[i,0], coords[i,1]), radius-3,shape=shape)
        # blank_canvas[rr, cc] = 0
    if random_shapes:
        radius = radius*5
        blank_canvas, labels = skim.draw.random_shapes(image_shape=shape,max_shapes=number,min_shapes=1,min_size=5,max_size=radius,channel_axis=None,intensity_range=(0,1))
        blank_canvas = 1-(blank_canvas/255)
    # print('created idealised image containing',number,'objects','with radius',radius)
    # print('blank_canvas.dtype',blank_canvas.dtype)
    return blank_canvas