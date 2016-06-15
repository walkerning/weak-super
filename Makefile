all:
	python setup.py develop --user
	python setup.py build_ext --inplace
