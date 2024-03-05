#!/bin/bash


for i in {0..6}
do
	python3 analysisTest.py $i &
done