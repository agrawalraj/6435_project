#!/bin/sh
for i in $(seq 0 10 2000)
do
	for n_induce in 50 100 200 500
	do
	  echo "'At iteration $i for inducing $n_induce"
	  ipython main_effects_bad.py $n_induce $i
	done
done