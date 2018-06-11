# -*- coding: utf-8 -*-
import argparse 
parser = argparse.ArgumentParser(prog='demo', description='A demo program', epilog='The end of usage')

parser.add_argument('name')
parser.add_argument('-a', '--age', type=int, required=True)
parser.add_argument('-s', '--status', choices=['alpha', 'beta', 'released'], type=str, dest='myStatus')

args = parser.parse_args() # 将名字空间赋值给args
print(args) # 输出名字空间