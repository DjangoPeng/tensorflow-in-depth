# -*- coding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser(prog='demo')
parser.add_argument('name')
parser.add_argument('-a', '--age', type=int, required=True)
parser.add_argument('-s', '--status', choices=['alpha', 'beta', 'released'], type=str, dest='myStatus')
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

args, unparsed = parser.parse_known_args() # 将解析器中未定义的参数返回给unparsed
print('args=%s, unparsed=%s' % (args, unparsed))