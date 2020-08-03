import uuid
from flask import jsonify, abort, request, Blueprint
import os
import planning_logic.icpcp_greedy_repair_cycle as icpcp_greedy_repair
import networkx as nx
import random as rng
import sys
REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


def prepare_icpcp(dependencies, tasks, performance_model=None):
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

    number_of_nodes = G.number_of_nodes()
    t = 0
    for line in range(0, len(performance_model)):
        t += 1
        tstr = "time" + str(t)
        for inode in range(0, number_of_nodes):
            G.node[inode][tstr] = performance_model[line][inode]

    print(G.number_of_nodes())
    return G

@REQUEST_API.route('/plan', methods=['POST'])
def send_vm_configuration():
    """Return optimal vm configuration
    """

    # if not request.files['file']:
    #     abort(400)

    data = request.get_json(force=True)
    dependencies = data['dependencies']
    tasks = data['tasks']
    icpcp_params = data['icpcp_params']
    performance = price = icpcp_params['performance']
    price = icpcp_params['price']
    deadline = icpcp_params['deadline']
    # endpoint_parameters = data[]
    graph = prepare_icpcp(dependencies, tasks, performance)
    icpcp_greedy_repair.main(sys.argv[1:], command_line=False, graph=graph, prep_prices=price, prep_deadline=deadline)
    print(icpcp_greedy_repair.number_of_nodes)
    print(icpcp_greedy_repair.instances)

    # TODO: handle error codes
    return jsonify(data), 200


if __name__ == '__main__':
    print(os.getcwd())
    icpcp_greedy_repair.main(sys.argv[1:])
    print(icpcp_greedy_repair.number_of_nodes)
    print(icpcp_greedy_repair.instances)
