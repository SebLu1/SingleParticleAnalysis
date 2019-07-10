import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--foo', help='Description for foo argument',
                    required=True)
parser.add_argument('--bar', help='Description for bar argument',
                    required=True)
args = vars(parser.parse_args())

print(type(args))
print(args)
print(args['foo'])
print(args['bar'])
