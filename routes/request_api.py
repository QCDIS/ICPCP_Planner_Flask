import uuid
from flask import jsonify, abort, request, Blueprint
import os
from cwlparser import CwlParser
REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API





@REQUEST_API.route('/plan', methods=['POST'])
def send_vm_configuration():
    """Return optimal vm configuration
    """

    if not request.files['file']:
        abort(400)

    data = request.get_json(force=True)

    # TODO: handle error codes
    return jsonify(data), 200

