import argparse


def get_args():
    # args
    # used for args : https://pymotw.com/3/argparse/
    parser = argparse.ArgumentParser(
        description='Predict flower names from images',
    )

    parser.add_argument('image_path', action="store")
    parser.add_argument('model', action="store")
    parser.add_argument('--top_k', action="store", dest='top_k', type=int)
    parser.add_argument('--category_names', action="store", dest='category_names')

    return parser


class ArgumentClass:
    def __init__(self):
        args = get_args().parse_args()
        self.image_path = args.image_path
        self.model_path = args.model
        self.top_k = args.top_k
        if self.top_k is None:
            self.top_k = 1
        self.classes_path = args.category_names
