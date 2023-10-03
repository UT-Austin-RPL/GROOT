wget https://utexas.box.com/shared/static/zammwzfn7b5utqv6y94ra9nx0zc729fy.jpg

wget https://utexas.box.com/shared/static/18wy8v2fanqjzakl4b27y351626o6e0h.png

wget https://utexas.box.com/shared/static/v68ixuug2j4edg537dgunml8patv4x1e.hdf5

mkdir datasets
mkdir -p datasets/annotations

mv frame.jpg datasets/annotations
mv frame_annotation.png datasets/annotations
mv example_demo.hdf5 datasets/

