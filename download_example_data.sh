wget -O frame.jpg https://utexas.box.com/shared/static/zammwzfn7b5utqv6y94ra9nx0zc729fy.jpg

wget -O frame_annotation.png https://utexas.box.com/shared/static/18wy8v2fanqjzakl4b27y351626o6e0h.png

wget -O example_demo.hdf5 https://utexas.box.com/shared/static/v68ixuug2j4edg537dgunml8patv4x1e.hdf5

wget -O example_new_object.jpg https://utexas.box.com/shared/static/2eky71626yqr71mgfg0z1w40hfzfekj9.jpg

mkdir datasets
mkdir -p datasets/annotations/example_demo

mv frame.jpg datasets/annotations/example_demo
mv frame_annotation.png datasets/annotations/example_demo
mv example_demo.hdf5 datasets/
mv example_new_object.jpg datasets/annotations/example_demo
