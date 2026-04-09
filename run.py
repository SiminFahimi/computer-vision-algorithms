import argparse

from harris_corner.demo import run_harris
from canny_edge.demo import run_canny
from laplacian_pyramid.demo import run_pyramid
from object_detection_hog.demo import run_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)

    args = parser.parse_args()

    if args.method == "harris":
        run_harris()

    elif args.method == "canny":
        run_canny()

    elif args.method == "pyramid":
        run_pyramid()

    elif args.method == "detect":
        run_detector()

    else:
        print("Unknown method")


if __name__ == "__main__":
    main()
