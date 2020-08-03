import planning_logic.icpcp_greedy_repair_cycle as icpcp_greedy_repair
import sys

if __name__ == '__main__':
    icpcp_greedy_repair.main(sys.argv[1:])
    nodes_in_inst = 0
    number_of_nodes = icpcp_greedy_repair.number_of_nodes
    G = icpcp_greedy_repair.G
    instances = icpcp_greedy_repair.instances
    rstr = "\nPCP solution for task graph with " + str(number_of_nodes) + " nodes"
    rstr += "\n     Start time    Stop time    Duration    Inst cost    Number of nodes"
    for inst in instances:
        if len(inst) > 0:
            linst = len(inst)
            nodes_in_inst += linst
            serv = G.node[inst[0]]["Service"]
            ninst = G.node[inst[0]]["Instance"]
            est = G.node[inst[0]]["EST"]
            eft = G.node[inst[linst - 1]]["EFT"]
            duration = eft - est
            rstr += "\nS" + str(serv) + "," + str(ninst)
            rstr += "   " + str(est) + "    " + str(eft) + "    " + str(duration)

    tasklist = ""
    for k in range(0, linst):
        if k > 0:
            tasklist += ", "
        tasklist += G.node[inst[k]]["name"]
    rstr += "    " + tasklist
    print(icpcp_greedy_repair.number_of_nodes)
    print(icpcp_greedy_repair.instances)
    print(rstr)