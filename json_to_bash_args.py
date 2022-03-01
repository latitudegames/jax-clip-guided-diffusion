import json
import sys

args = json.loads(sys.argv[1])

for key, val in args.items():
    print(f'--{key}')
    if type(val) is list:
        for item in val:
            print(item)
    else:
        print(val)
