# MPC-learning
MPC-learning is a Python library for performing multi-party computation on machine learning applications. This library implements the 3-party computation protocol of https://eprint.iacr.org/2016/768.pdf . For now, a "dealer" is required to distribute shares of inputs, and the protocol can only be run locally (does not support networking yet).

## Installation
This is a quick guide to getting this repo up and running for development.

0. Clone the library

    ```bash
    $ git clone https://github.com/trailofbits/mpc-learning
    ```

1. Download virtualenv.

2. Create and source your virtual environment.

    ```bash
    $ virtualenv -p python3 .venv
    $ source .venv/bin/activate
    ```

3. Install the library:

    a. if you want to use and edit the library:
    ```bash
    $ python setup.py develop
    ```

    b. otherwise to just use the library:
    ```bash
    $ python setup.py install
    ```

## Usage

If everything installed correctly the following examples should work:

1. raw perceptron algorithm:

    ```bash
    $ python examples/perceptron/alg.py
    ```

2. mpc perceptron (should be same result as raw algorithm, but will take longer):

    ```bash
    $ python examples/perceptron/eval_circuit.py
    ```

3. raw svm algorithm:

    ```bash
    $ python examples/svm/alg.py
    ```

4. mpc svm (should be same result as raw algorithm, but will take longer):

    ```bash
    $ python examples/svm/eval_circuit.py
    ```

If you would like to run this library on a different algorithm, you will have to synthesize the corresponding circuit for one iteration of the algorithm. The circuits must be in the correct format. For reference, checkout the perceptron and svm circuits: examples/*/circuit.py

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)