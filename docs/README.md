# Documentation

## Dependencies

To build document it requires [Doxygen](https://www.doxygen.nl/) and [Sphinx](https://www.sphinx-doc.org/).

To install Doxygen, please turn to its official site to find instructions.

To install Sphinx, please use `pip` and the [requirements file](./requirements.txt) to install dependencies.

```bash
pip install -r requirements.txt
```

## Build

Use the `make` command to build the document which is followed by the target format, such as `html`.

```bash
make html
```

Then the document will appear at the `/docs/_build` directory.

## Internationalization

Please find instructions in the README file for each language other than English.

- [Chinese Simplified](./README.zh_CN.md)
