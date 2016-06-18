all:
	python setup.py develop --user
	python setup.py build_ext --inplace

clean-pyc:
	find . -name '*.pyc' -type f | xargs -I{} rm {}
