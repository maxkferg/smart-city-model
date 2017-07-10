# Camera Mapping

In this directory we map camera footage from a `camera` to coordinates
in an `environment`.

The workflow is like this:
* Specify reference points on camera image
* Map camera image to birds eye view
* Save configuration file for external use

# Example

```python

from birdseye import BirdseyeView

view = Birdseye("auburn")

# Return pts in the global (birdseye) coordinate system
view.transform_to_global(pts)

# Return the original image as a numpy array
view.get_original_image()

# Return the birdseye image as a np array
view.get_transformed_image()
```

# BirdseyeView Class

## BirdseyeView\#transform_to_global(pts)
Map an array of local points to global coordinates so they can be plotted on the birdseye image
The top left corner is considered the origin
@pts. A nx2 matrix, where each row is a point

## BirdseyeView\#get_original_image()
Return a single camera frame

## BirdseyeView\#get_transformed_image()
Return a single camera frame that has been transformed into global coordinates

