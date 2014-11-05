3 SCENES:

- diffuse.rayTracing
- my_diffuse.ppm (normal code result)
- diffuse_without_419.ppm (same as last, except line 419 was commented out)

- scenell.rayTracing
- scenell.ppm (normal code, but in my code I made a tweak that makes the image look better even though the tweak was a hacky work-around)

- my_scene.rayTracing
- my_scene.ppm (I used triangles and a reflective sphere to show phong, diffuse, ambient, specular, and shadowing. Only missing refraction).


MISC.
- My code is mostly working, although a few bugs still remain. Tonight while working I fixed another bug, on line 269 I had "t = t1" instead of "t = t0".
- The only part of the project I wasn't able to get to was the Refraction, and bug fixing. There are bugs, but I feel good about how everything turned out.


HOW TO RUN:
Line 497: Enter the file you want to read, and the height/width of the image (note: all images are square)
Line 493: Enter the name of what you want the output ppm file to be called
In command line: "python Raytracer.py"


Let me know if you have any questions.