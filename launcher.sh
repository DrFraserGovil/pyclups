#!/bin/bash


for i in {0..6}
do
	python analysisTest.py $i &
done