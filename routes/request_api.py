import uuid
from flask import jsonify, abort, request, Blueprint
import os
import planning_logic.icpcp_greedy_repair as icpcp_greedy_repair
import networkx as nx
import random as rng
import sys
from planning_logic.instance import Instance as vm_instance
from planning_logic.icpcp_greedy import Workflow

REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


def prepare_icpcp(dependencies, tasks, performance_model=None):
    G = nx.DiGraph()
    # add tasks to graph
    for i in range(0, len(tasks)):
        if not G.has_node(i):
            G.add_node(i)
            G.node[i]["order"] = i
            G.node[i]["name"] = tasks[i]
            G.node[i]["time1"] = 0
            G.node[i]["time2"] = 0
            G.node[i]["time3"] = 0

    # add dependencies
    for key, value in dependencies.items():
        for edge_node in value:
            # TODO: find better way to do this, as this will slow our program
            key_index = tasks.index(key)
            edge_node_index = tasks.index(edge_node)
            # throughput = rng.randrange(0, 5)
            throughput = 0
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


def prepare_icpcp_greedy(tasks, dependencies):
    graph = nx.DiGraph()

    for i in range(0, len(tasks)):
        task = tasks[i]
        graph.add_node(i, order=i, name=task, est=-1, eft=-1, lst=-1,
                       lft=-1)

    for key, value in dependencies.items():
        for edge_node in value:
            # throughput = rng.randrange(0, 5)
            throughput = 0
            key_index = tasks.index(key)
            edge_node_index = tasks.index(edge_node)
            graph.add_weighted_edges_from([(key_index, edge_node_index, throughput)])

    return graph


def run_icpc_greedy(dag, performance, price, deadline):
    wf = Workflow()
    print(os.getcwd())
    wf.init(dag, performance, price, deadline)
    wf.calc_startConfiguration(-1)

    start_cost, start_eft = wf.getStartCost()
    start_str = "start configuartion: cost=" + str(start_cost) + "  EFT(exit)=" + str(start_eft)
    print("\nStart situation")
    wf.printGraphTimes()

    wf.ic_pcp()
    print("\nEnd situation")
    wf.printGraphTimes()

    # vm and exit node not part of PCP, so
    # adjust LST, LFT of vm node
    # adjust EST, EFT of exit node
    wf.update_node(0)
    wf.update_node(wf.number_of_nodes() - 1)
    #
    # # check PCP end situation
    wf.updateGraphTimes()

    retVal = wf.checkGraphTimes()
    print("checkGraphTimes: retVal=" + str(retVal))
    tot_idle = wf.checkIdleTime()
    print("checkIdleTime: idle time=" + str(tot_idle))

    wf.print_instances(tot_idle)

    print("\n" + start_str)
    if retVal == -1:
        print("\n**** Invalid final configuration ****")
    else:
        final_cost, final_eft = wf.cal_cost()
        print("final configuration: cost=" + str(final_cost) + "  EFT(exit)=" + str(final_eft))

    return wf.instances


@REQUEST_API.route('/plan', methods=['POST'])
def send_vm_configuration():
    """Return optimal vm configuration
    """

    # if not request.files['file']:
    #     abort(400)

    # set to false to use greedy version of icpcp
    greedy_repair = True

    # extract data from request
    data = request.get_json(force=True)
    dependencies = data['dependencies']
    tasks = data['tasks']
    icpcp_params = data['icpcp_params']
    performance = icpcp_params['performance']
    price = icpcp_params['price']
    deadline = icpcp_params['deadline']

    if (greedy_repair):
        # put parameters in a graph to be able to run icpcp
        graph = prepare_icpcp(dependencies, tasks, performance)
        icpcp_greedy_repair.main(sys.argv[1:], command_line=False, graph=graph, prep_prices=price,
                                 prep_deadline=deadline)

        # now we want to extract the number of vms and other relevant data
        nodes_in_inst = 0
        number_of_nodes = icpcp_greedy_repair.number_of_nodes
        G = icpcp_greedy_repair.G
        instances = icpcp_greedy_repair.instances

        servers = []
        for inst in instances:
            if len(inst) > 0:
                linst = len(inst)
                nodes_in_inst += linst
                serv = G.node[inst[0]]["Service"]
                ninst = G.node[inst[0]]["Instance"]
                est = G.node[inst[0]]["EST"]
                eft = G.node[inst[linst - 1]]["EFT"]
                duration = eft - est

                tasklist = []
                for k in range(0, linst):
                    tasklist.append(G.node[inst[k]]["name"])

                server = vm_instance(serv, duration, est, eft, tasklist)
                servers.append(server)

    else:
        graph = prepare_icpcp_greedy(tasks, dependencies)
        servers = run_icpc_greedy(graph, performance, price, deadline)

    # put relevant extracted data in json format to be sent back to the backend
    response_json = []

    for i in range(0, len(servers)):
        instance = servers[i]
        x = {'num_cpus': i + 1, 'disk_size': "{} GB".format((i + 1) * 10),
             'mem_size': "{} MB".format(int((i + 1) * 4096))}
        instance.properties = x
        if not greedy_repair:
            for t in instance.task_list:
                instance.task_names.append(tasks[t - 1])

    # generate more output format
    for serv in servers:

        entry = serv.properties
        entry['tasks'] = serv.task_names
        if not greedy_repair:
            entry['vm_start'] = serv.vm_start.item()
            entry['vm_end'] = serv.vm_end.item()
            entry['vm_cost'] = serv.vm_cost
            entry['vm_type'] = serv.vm_type
        else:
            entry['vm_start'] = serv.vm_start
            entry['vm_end'] = serv.vm_end
            entry['vm_cost'] = price[serv.vm_type - 1]
            entry['vm_type'] = serv.vm_type

        response_json.append(entry)

    print(response_json)
    # TODO: handle error codes
    return jsonify(response_json), 200


if __name__ == '__main__':
    print(os.getcwd())
    icpcp_greedy_repair.main(sys.argv[1:])
    print(icpcp_greedy_repair.number_of_nodes)
    print(icpcp_greedy_repair.instances)
