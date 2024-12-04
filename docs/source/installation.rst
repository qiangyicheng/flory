Installation
============

Install using pip
-----------------

The :mod:`flory` Python package is available on `pypi <https://pypi.org/project/flory/>`_, so you should be able to install it by running

.. code-block:: bash

   pip install flory


By default, :mod:`flory` package only comes with the dependencies directly referenced by its main functionality. The dependencies for examples, tests and documentations are not included. To install all these dependencies, you can specify a :code:`dev` option:

.. code-block:: bash

   pip install 'flory[dev]'


You may also want to select among the optional dependencies for examples, tests or documentations. In this case, you can specify option :code:`example`, :code:`test` or :code:`doc`. For example:

.. code-block:: bash

   pip install 'flory[example]'


Install using conda
-------------------

As an alternative, the :mod:`flory` Python package is also available through `conda <https://conda.io>`_ using the [conda-forge](https://conda-forge.org/) channel:

.. code-block:: bash

   conda install -c conda-forge flory


If you are using conda, we recommend that you install the optional dependencies directly:

.. code-block:: bash

   conda install -c conda-forge --file https://raw.githubusercontent.com/qiangyicheng/flory/main/examples/requirements.txt
   conda install -c conda-forge --file https://raw.githubusercontent.com/qiangyicheng/flory/main/tests/requirements.txt
   conda install -c conda-forge --file https://raw.githubusercontent.com/qiangyicheng/flory/main/docs/requirements.txt


Test installation
-----------------

If the optional dependencies for tests are installed, you can run tests in root directory of the package with :code:`pytest`. By default, some slow tests are skipped. You can run them with the :code:`--runslow` option:

.. code-block:: bash

   pytest --runslow

