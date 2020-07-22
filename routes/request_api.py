import uuid
from flask import jsonify, abort, request, Blueprint
import os
from cwlparser import CwlParser
import networkx as nx
import random as rng
REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


def prepare_icpcp(dependencies, tasks):
    G = nx.DiGraph()
    #add tasks to graph
    for i in range(0, len(tasks)):
        if not G.has_node(i):
            G.add_node(i)
            G.node[i]["order"] = i
            G.node[i]["name"] = tasks[i]
            G.node[i]["time1"] = 0
            G.node[i]["time2"] = 0
            G.node[i]["time3"] = 0

    #add dependencies
    for key, value in dependencies.items():
        for edge_node in value:
            #TODO: find better way to do this, as this will slow our program
            key_index = tasks.index(key)
            edge_node_index = tasks.index(edge_node)
            throughput = rng.randrange(0, 5)
            G.add_edge(key_index, edge_node_index)
            G[key_index][edge_node_index]['throughput'] = throughput

    print(G.number_of_nodes())


@REQUEST_API.route('/plan', methods=['POST'])
def send_vm_configuration():
    """Return optimal vm configuration
    """

    # if not request.files['file']:
    #     abort(400)

    data = request.get_json(force=True)
    dependencies = data['dependencies']
    tasks = data['tasks']
    prepare_icpcp(dependencies, tasks)
    # TODO: handle error codes
    return jsonify(data), 200

