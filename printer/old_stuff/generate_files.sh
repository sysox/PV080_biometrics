#!/bin/sh 

python3 gendocx.py -p $1
lowriter --convert-to pdf ./print.docx

