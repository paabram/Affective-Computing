To run the experiments, call: python Project1.py -<data type> -<data directory>
    <data type>: One of o (original data points), t (translated to origin), x, y, or z (rotated on that axis 180 degrees). This argument may be omitted to run all 5 methods.
    <data directory>: Path to facial landmarks data, structured as parent directory > subject folders > emotion folders > data files

Required packages to install:
- numpy
- scikit-learn
- matplotlib