PROJECT=bitwave

help:
	@echo "make clean package upload install unittest pyright coverage mypy"

package:
	@python3 -m build

install:
	@python3 -m pip install -e .

uninstall:
	@python3 -m pip uninstall $(PROJECT)

check:
	@python3 -m twine check dist/*

upload:
	@python3 -m twine upload --repository pypi dist/* --verbose

clean:
	@python3 ./clean.py

unittest:
	@python3 -m unittest discover

pyright:
	pyright

coverage:
	pytest --cov=$(PROJECT) --cov-report=xml

mypy:
	mypy --show-traceback .
