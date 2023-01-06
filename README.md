# Image Morphing
This project aims to generate a `.mp4` file of the transition from `Image1` to `Image2`. First, the user should choose the corresponding points in two images. Here is an example of selecting corresponding points between two images:
<p align="center">
<img height = "300" src="https://user-images.githubusercontent.com/46090276/211057899-e44fc143-e6fb-422b-89e8-520dbc7bf2a1.JPG" alt="Corresponding_Points">
</p>

Then, the Delaunay triangularization algorithm is used to partition images into small triangles. 
<p align="center">
<img height = "300" src="https://user-images.githubusercontent.com/46090276/211058837-66b2d541-2360-4f50-b687-421f2beab65e.JPG" alt="Delaunay">
</p>

We generate each video frame based on image warping and the convex combination of pixels' intensities. Here is my result:

<p align="center">
<img src="https://user-images.githubusercontent.com/46090276/211064186-316c89f5-936b-47cf-9ab4-61fe215f831a.gif" alt="gif")
</p>

