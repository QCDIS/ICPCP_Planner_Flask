"""The tests to run in this project.
To run the tests type,
$ nosetests --verbose
"""

from nose.tools import assert_true
import os
import requests

BASE_URL = "http://localhost:5000"
DIR_TEST_FILES = os.path.join(os.getcwd(), "test_files")


def test_parse_file():
    "Test sending cwl file to parser "

    cwl_file = os.path.join(DIR_TEST_FILES, "compile1.cwl")
    files = {'file': open(cwl_file, 'rb')}

    response = requests.post('%s/send_file' % (BASE_URL), files=files)
    assert_true(response.status_code == 200)


