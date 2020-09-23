'''
Created on Nov 20, 2015

@author: junchao
'''

import os
import sys
import re
import random
import networkx as nx
import numpy as np
import json
import yaml
import io
from planning_logic.vm_instance_icpcp import NewInstance
from subprocess import call
import time
from collections import deque

from optparse import OptionParser


# from __builtin__ import True
from networkx import DiGraph


class Workflow:


    def init(self, dag, performance, price, deadline):

        # Initialization
        self.visited = []
        self.G = nx.DiGraph()
        self.vertex_num = 0
        self.successful = 0
        self.deadline = 0

        #load input
        self.G = dag
        self.vertex_num = len(dag.nodes())

        self.vm_price = price
        self.deadline = deadline
        perf_data = performance
        l = []
        for vm_performance in perf_data:
            l.append(vm_performance)
        # with open(combined_input, 'r') as stream:
        #     data_loaded = yaml.safe_load(stream)
        #     self.vm_price = data_loaded[0]["price"]
        #     self.deadline = data_loaded[2]["deadline"]
        #     perf_data = data_loaded[1]["performance"]
        #     l = []
        #     for key, value in perf_data.items():
        #         l.append(value)



        # print performance table
        self.p_table = np.matrix(l)
        print("number of nodes in G: " + str(self.vertex_num))
        print(self.p_table)

        # two reasons for adding entry node
        # a) no entry node present
        # b) current entry node has non-zero performance
        inlist = list(self.G.in_degree())
        print(inlist)
        num_zero = 0
        for j in inlist:
            if j[1] == 0:
                num_zero += 1
        if num_zero > 1 or (num_zero == 1 and self.p_table[0, 0] > 0):
            if num_zero > 1:
                print("Add entry node to graph; dag file has no entry node")
            else:
                print("Add entry node to graph; dag file has entry node with non-zero performance")
            self.add_entry_node()
            # adjust perf table
            self.p_table = np.insert(self.p_table, 0, np.zeros((1, 1), dtype=int), axis=1)
        print("number of nodes in G: " + str(self.vertex_num))
        print(self.p_table)

        # two reasons for adding exit node
        # a) no exit node present
        # b) current exit node has non-zero performance
        outlist = list(self.G.out_degree())
        num_zero = 0
        for j in outlist:
            if j[1] == 0:
                num_zero += 1
        if num_zero > 1 or (num_zero == 1 and self.p_table[0, self.vertex_num - 1] > 0):
            if num_zero > 1:
                print("Add exit node to graph; dag file has no exit node")
            else:
                print("Add exit node to graph; dag file has exit node with non-zero performance")
            self.add_exit_node()
            # sys.exit("\nQuit\n")
            # adjust perf table
            self.p_table = np.append(self.p_table, np.zeros((3, 1), dtype=int), axis=1)
        print("number of nodes in G: " + str(self.vertex_num))
        print(self.p_table)

        self.assigned_list = [-1] * (self.vertex_num)



        self.instances = []
        if len(list(nx.simple_cycles(self.G))) > 0:
            print('the DAG contains cycles!')
        d_list = []






    def add_entry_node(self):
        # copy grap
        G1 = nx.DiGraph()
        for u in self.G.nodes():
            unum = u + 1
            uname = "t" + str(unum)
            uorder = unum
            G1.add_node(u + 1, order=uorder, name=uname, est=-1, eft=-1, lst=-1, lft=-1)
        for u, v in self.G.edges():
            G1.add_weighted_edges_from([(u + 1, v + 1, self.G[u][v]['weight'])])

        print("Add entry node to graph")
        name = "t" + str(0)
        G1.add_node(0, order=0, name=name, est=-1, eft=-1, lst=-1, lft=-1)
        endnodes = []
        for vertex in G1.nodes():
            if G1.in_degree(vertex) == 0 and not vertex == 0:
                endnodes.append(vertex)

        for node in endnodes:
            G1.add_weighted_edges_from([(0, node, 0)])
        self.G = G1
        self.vertex_num += 1

    def add_exit_node(self):

        print("Add exit node to graph")
        name = "t" + str(self.vertex_num)
        self.G.add_node(self.vertex_num, order=self.vertex_num - 1, name=name, est=-1, eft=-1, lst=-1, lft=-1)
        startnodes = []
        for vertex in self.G.nodes():
            if self.G.out_degree(vertex) == 0 and not vertex == self.vertex_num:
                startnodes.append(vertex)

        for node in startnodes:
            self.G.add_weighted_edges_from([(node, self.vertex_num, 0)])

        self.vertex_num += 1

    def number_of_nodes(self):
        return self.vertex_num

    def calc_startConfiguration(self, perc_deadline):
        self.G.node[0]['est'] = 0
        self.G.node[0]['eft'] = self.p_table[0, 0]
        self.cal_est(0)
        self.G.node[self.vertex_num - 1]['lft'] = int(self.deadline)
        self.G.node[self.vertex_num - 1]['lst'] = int(self.deadline) - self.p_table[0, self.vertex_num - 1]
        self.cal_lft(self.vertex_num - 1)
        # self.updateGraphTimes()

        pcp = []
        d = self.vertex_num - 1
        di = d
        while True:
            cp = self.getCriticalParent(di)
            if cp == -1:
                break
            pcp = [cp] + pcp
            di = cp
        pcp.append(d)
        if len(pcp) == 0:
            sys.exit("\nERROR **** No critical path found ****\n")
        criticalpath_time = 0
        for j in range(0, len(pcp) - 1):
            criticalpath_time += (self.G.node[pcp[j]]["eft"] - self.G.node[pcp[j]]["est"])
            throughput = self.G[pcp[j]][pcp[j + 1]]["weight"]
            criticalpath_time += throughput
        criticalpath_time += (self.G.node[pcp[len(pcp) - 1]]["eft"] - self.G.node[pcp[len(pcp) - 1]]["est"])
        print("\nTime critical path:" + str(criticalpath_time))
        if perc_deadline > 0:
            self.deadline = int(100. * float(criticalpath_time) / float(perc_deadline))
            print("New deadline: ", self.deadline)
            self.G.node[0]['est'] = 0
            self.G.node[0]['eft'] = self.p_table[0, 0]
            self.cal_est(0)
            self.G.node[self.vertex_num - 1]['lft'] = int(self.deadline)
            self.G.node[self.vertex_num - 1]['lst'] = int(self.deadline) - self.p_table[0, self.vertex_num - 1]
            self.cal_lft(self.vertex_num - 1)
        else:
            print("Deadline: ", self.deadline)

    def getCriticalParent(self, d):

        max_time = 0
        cp = -1
        d_est = self.G.node[d]["est"]
        p_iter = iter(self.G.predecessors(d))
        while True:
            try:
                p = next(p_iter)

                ctime = self.G.node[p]["eft"]
                ctime += self.G[p][d]["weight"]
                if ctime >= max_time:
                    max_time = ctime
                    cp = p
            except StopIteration:
                break
        return cp

        # The following two functions are to initialize the EST, EFT and LFT

    # calculate the earliest start time and earliest finish time
    def cal_est(self, i):
        successors = []
        c_iter = iter(self.G.successors(i))
        while True:
            try:
                c = next(c_iter)
                successors.append(c)
            except StopIteration:
                break
        for child in successors:
            est = self.G.node[i]['eft'] + self.G[i][child]['weight']
            if est > self.G.node[child]['est']:
                self.G.node[child]['est'] = est
                eft = est + self.p_table[0, child]
                self.G.node[child]['eft'] = eft
                # print self.G.node[child]['name']+":est="+str(est)+",eft="+str(eft)
                self.cal_est(child)

    def cal_lft(self, d):
        predecessors = []
        c_iter = iter(self.G.predecessors(d))
        while True:
            try:
                p = next(c_iter)
                predecessors.append(p)
            except StopIteration:
                break
        for parent in predecessors:
            lft = self.G.node[d]['lft'] - self.p_table[0, d] - self.G[parent][d]['weight']
            if self.G.node[parent]['lft'] == -1 or lft < self.G.node[parent]['lft']:
                # parent may not finish later
                self.G.node[parent]['lft'] = lft
                lst = lft - self.p_table[0, parent]
                self.G.node[parent]['lst'] = lst
                # print "call graphAssignLFT(",graph.node[parent]['name'],")"
                self.cal_lft(parent)

                # Finding critical path

    def ic_pcp(self):
        self.assigned_list[0] = 0
        self.assigned_list[self.vertex_num - 1] = 0
        self.assign_parents(self.vertex_num - 1)

    def has_unassigned_parent(self, i):
        predecessors = []
        c_iter = iter(self.G.predecessors(i))
        while True:
            try:
                p = next(c_iter)
                predecessors.append(p)
            except StopIteration:
                break
        for parent in predecessors:
            if (self.assigned_list[parent] == -1):
                return True
        return False

    def assign_parents(self, i):
        while (self.has_unassigned_parent(i)):
            if self.successful == 1:  # resources cannot be met
                break
            pcp = []
            self.find_critical_path(i, pcp)
            print("critical path:", pcp)
            assigned_vm = self.assign_path(pcp)
            if not assigned_vm == -2 and not assigned_vm == -1:
                self.G.node[pcp[len(pcp) - 1]]['eft'] = self.G.node[pcp[len(pcp) - 1]]['est'] + self.p_table[
                    self.assigned_list[pcp[len(pcp) - 1]], pcp[len(pcp) - 1]]
                self.G.node[pcp[0]]['lst'] = self.G.node[pcp[0]]['lft'] - self.p_table[
                    self.assigned_list[pcp[0]], pcp[0]]
                # save the assigned VM in the instance list
                # TODO: Decision of the time slot for the new instance
                ni = NewInstance(assigned_vm, self.vm_price[assigned_vm], self.G.node[pcp[len(pcp) - 1]]['est'], self.G.node[pcp[0]]['eft'], pcp)
                self.instances.append(ni)
            elif assigned_vm == -1:
                print("the available resources cannot meet the deadline with the IC-PCP algorithm")
                self.successful = 1
                break
                # sys.exit()
            else:
                self.update_est(pcp[len(pcp) - 1], pcp)
                self.update_lft(pcp[0], pcp)
            for j in reversed(pcp):  # also in the paper they didn't mention the order
                self.assign_parents(j)

    # TODO: A very tricky thing on updating the EST and EFT.

    def update_est(self, i, pcp):
        for child in self.G.successors(i):
            if child not in pcp:
                est = self.G.node[i]['eft'] + self.G[i][child]['weight']
                if self.assigned_list[i] == -1:
                    eft = est + self.p_table[0, child]
                else:
                    eft = est + self.p_table[self.assigned_list[i], child]
            else:
                est = self.G.node[i]['eft']
                eft = est + self.p_table[self.assigned_list[i], child]

            # decide whether the assignment will violate other parent data dependency
            all_smaller = True
            for parent in self.G.predecessors(child):
                if not parent == i:
                    if self.G.node[parent]['eft'] + self.G[parent][child]['weight'] > est:
                        all_smaller = False
            if all_smaller:
                self.G.node[child]['est'] = est
                self.G.node[child]['eft'] = eft
                self.update_est(child, pcp)

    def update_lft(self, d, pcp):
        for parent in self.G.predecessors(d):
            if parent not in pcp:
                if self.assigned_list[d] == -1:
                    lft = self.G.node[d]['lft'] - self.p_table[0, d] - self.G[parent][d]['weight']
                else:
                    lft = self.G.node[d]['lft'] - self.p_table[self.assigned_list[d], d] - self.G[parent][d]['weight']
            else:
                if d in pcp:
                    lft = self.G.node[d]['lft'] - self.p_table[self.assigned_list[d], d]
                else:
                    lft = self.G.node[d]['lft'] - self.p_table[self.assigned_list[d], d] - self.G[parent][d]['weight']
            all_bigger = True
            for child in self.G.successors(parent):
                if not child == d:
                    if self.G.node[child]['lft'] - self.G[parent][child]['weight'] > lft:
                        all_bigger = False
            if all_bigger:
                self.G.node[parent]['lft'] = lft
                # print "call graphAssignLFT(",graph.node[parent]['name'],")"
                self.update_lft(parent, pcp)

    def update_est_old(self, i, pcp):
        for child in self.G.successors(i):
            if child not in pcp:
                est = self.G.node[i]['eft'] + self.G[i][child]['weight']
                # if self.assigned_list[i] == -1:
                if self.assigned_list[child] == -1:
                    eft = est + self.p_table[0, child]
                else:
                    # eft = est + self.p_table[self.assigned_list[i], child]
                    eft = est + self.p_table[self.assigned_list[child], child]
            else:
                est = self.G.node[i]['est'] + self.p_table[self.assigned_list[i], i]
                eft = est + self.p_table[self.assigned_list[child], child]

            # decide whether the assignment will violate other parent data dependency
            all_smaller = True
            for parent in self.G.predecessors(child):
                if not parent == i:
                    if self.G.node[parent]['eft'] + self.G[parent][child]['weight'] > est:
                        all_smaller = False
            if all_smaller:
                self.G.node[child]['est'] = est
                self.G.node[child]['eft'] = eft
                self.update_est(child, pcp)

    def update_lft_old(self, d, pcp):
        for parent in self.G.predecessors(d):
            if parent not in pcp:
                lft = self.G.node[d]['lst'] - self.G[parent][d]['weight']
                if self.assigned_list[parent] == -1:
                    lst = lft - self.p_table[0, parent]
                else:
                    lst = lft - self.p_table[self.assigned_list[parent], parent]
            else:
                lft = self.G.node[d]['lft'] - self.p_table[self.assigned_list[d], d]
                lst = lft - self.p_table[self.assigned_list[parent], parent]

            all_bigger = True
            for child in self.G.successors(parent):
                if not child == d:
                    if self.G.node[child]['lft'] - self.G[parent][child]['weight'] > lft:
                        all_bigger = False
            if all_bigger:
                self.G.node[parent]['lft'] = lft
                self.G.node[parent]['lst'] = lst
                self.update_lft(parent, pcp)

    def find_critical_path(self, i, pcp):
        cal_cost = 0
        critical_parent = -1
        for n in self.G.predecessors(i):
            if self.assigned_list[n] == -1:  # parent of node i is not assigned
                if self.G.node[n]['eft'] + self.G[n][i]['weight'] > cal_cost:
                    cal_cost = self.G.node[n]['eft'] + self.G[n][i]['weight']
                    critical_parent = n
        if not critical_parent == -1:
            pcp.append(critical_parent)
            self.find_critical_path(critical_parent, pcp)

    def exec_time_sum(self, pcp, vm_type):
        sum = 0
        for i in pcp:
            sum += self.p_table[vm_type, i]
        return sum

    # look forward one step when assigning a vm to a pcp how the est varies
    def est_vary(self, pcp, d):
        head_pcp = pcp[len(pcp) - 1]
        original_est = self.G.node[head_pcp]['est']
        biggest_est = -1
        biggest_parent = -1
        for parent in self.G.predecessors(head_pcp):
            if parent == d:
                est = self.G.node[parent]['eft']
            else:
                est = self.G.node[parent]['eft'] + self.G[parent][head_pcp]['weight']
            if biggest_est < est:
                biggest_est = est
                biggest_parent = parent
        return original_est - biggest_est

    # choose the best existing available instance for the pcp
    def choose_exist_instance(self, pcp):
        best_vm = None
        best_exec_time = -1
        best_vary_time = -1
        for vm in self.instances:
            head_pcp = pcp[len(pcp) - 1]
            for parent in self.G.predecessors(head_pcp):
                head_exist_pcp = vm.task_list[0]  # The last node of the previous critical path
                if parent == head_exist_pcp:
                    if best_vm == None:
                        best_vm = vm
                        exec_time = self.exec_time_sum(pcp, vm.vm_type)
                        best_exec_time = exec_time
                        best_vary_time = self.est_vary(pcp, head_exist_pcp)
                    else:
                        best_vm_head = vm.task_list[0]
                        # if assigned to the vm, what will the est be
                        exec_time = self.exec_time_sum(pcp, vm.vm_type)
                        # calculate the lft
                        varied_time = self.G.node[head_pcp]['est'] - self.est_vary(pcp, head_exist_pcp)
                        lft = varied_time + exec_time
                        if (exec_time - self.est_vary(pcp, head_exist_pcp)) * self.vm_price[vm.vm_type] > \
                                (best_exec_time - self.est_vary(pcp, best_vm.task_list[0])) * self.vm_price[
                            best_vm.vm_type] \
                                and lft < self.G.node[head_pcp]['lft']:  # also should not violate the lft
                            best_vm = vm
                            best_exec_time = exec_time
                            best_vary_time = varied_time
        if not best_vm == None:
            best_vm.vm_end = self.G.node[pcp[len(pcp) - 1]]['est'] - best_vary_time + best_exec_time
        return best_vm

    def assign_path(self, pcp):
        cheapest_vm = -1
        cheapest_sum = 9999999  # the initialized value should be a very large number
        chosen_instance = self.choose_exist_instance(pcp)
        if chosen_instance == None:  # no existing instance for the pcp
            for i in range(self.p_table.shape[
                               0]):  # use the the shape of the performance table to identify how many VM types are there
                violate_LFT = 0
                start = self.G.node[pcp[len(pcp) - 1]]['est']
                cost_sum = 0
                for j in range(len(pcp) - 1, -1, -1):
                    cost_sum += self.p_table[i, pcp[j]] * self.vm_price[i]
                    start = start + self.p_table[i, pcp[j]]
                    if self.G.node[pcp[j]]['lft'] < start:
                        violate_LFT = 1
                    # launch a new instance of the cheapest service which can finish each task of P before its LFT
                if violate_LFT == 0 and cost_sum < cheapest_sum:
                    cheapest_vm = i
                    cheapest_sum = cost_sum
            for i in range(len(pcp)):
                self.assigned_list[pcp[i]] = cheapest_vm
            # adjust est pcp
            est = self.G.node[pcp[len(pcp) - 1]]['est']
            self.G.node[pcp[len(pcp) - 1]]['eft'] = est + self.p_table[cheapest_vm, pcp[len(pcp) - 1]]
            for j in range(len(pcp) - 2, -1, -1):
                est = self.G.node[pcp[j + 1]]['eft']
                self.G.node[pcp[j]]['est'] = est
                self.G.node[pcp[j]]['eft'] = est + self.p_table[cheapest_vm, pcp[j]]
            # adjust lft pcp
            lft = self.G.node[pcp[0]]['lft']
            self.G.node[pcp[0]]['lst'] = lft - self.p_table[cheapest_vm, pcp[0]]
            for j in range(1, len(pcp)):
                lft = self.G.node[pcp[j - 1]]['lst']
                self.G.node[pcp[j]]['lft'] = lft
                self.G.node[pcp[j]]['lst'] = lft - self.p_table[cheapest_vm, pcp[j]]
            return cheapest_vm
        else:  # found an instance that
            chosen_instance.task_list += pcp
            for i in range(len(pcp)):
                self.assigned_list[pcp[i]] = chosen_instance.vm_type
            # update est and eft of the pcp head node
            head_pcp = pcp[len(pcp) - 1]
            biggest_est = -1
            for parent in self.G.predecessors(head_pcp):
                est = 0
                if parent in chosen_instance.task_list:
                    est = self.G.node[parent]['eft']
                else:
                    est = est + self.G[parent][head_pcp]['weight']
                if biggest_est < est:
                    biggest_est = est
            self.G.node[head_pcp]['est'] = biggest_est
            self.G.node[head_pcp]['eft'] = self.G.node[head_pcp]['est'] + self.p_table[
                self.assigned_list[head_pcp], head_pcp]
            est = self.G.node[pcp[len(pcp) - 1]]['est']
            self.G.node[pcp[len(pcp) - 1]]['eft'] = est + self.p_table[chosen_instance.vm_type, pcp[len(pcp) - 1]]
            for j in range(len(pcp) - 2, -1, -1):
                est = self.G.node[pcp[j + 1]]['eft']
                self.G.node[pcp[j]]['est'] = est
                self.G.node[pcp[j]]['eft'] = est + self.p_table[chosen_instance.vm_type, pcp[j]]
            # adjust lft pcp
            lft = self.G.node[pcp[0]]['lft']
            self.G.node[pcp[0]]['lst'] = lft - self.p_table[chosen_instance.vm_type, pcp[0]]
            for j in range(1, len(pcp)):
                lft = self.G.node[pcp[j - 1]]['lst']
                self.G.node[pcp[j]]['lft'] = lft
                self.G.node[pcp[j]]['lst'] = lft - self.p_table[chosen_instance.vm_type, pcp[j]]

            # update the resource reservation time

            return -2

    def generate_string(self, node):
        s = "name " + str(node) + "\n"
        for i in range(len(self.vm_price)):
            s = s + str(self.p_table[i, node]) + "\n"
        s = s + "assigned vm: " + str(self.assigned_list[node] + 1)
        return s

    def update_instances(self):
        for vm in self.instances:
            vm_tasklist = vm.task_list
            begin = self.G.node[vm_tasklist[len(vm_tasklist) - 1]]["est"]
            end = self.G.node[vm_tasklist[0]]["eft"]
            vm.vm_start = begin
            vm.vm_end = end

    # calculate the total execution cost
    def cal_cost(self):
        cost = 0
        for vm in self.instances:
            cost = cost + self.vm_price[vm.vm_type] * (vm.vm_end - vm.vm_start)
        return (cost, self.G.node[self.vertex_num - 1]["eft"])

    def has_edge_vm(self, vm1, vm2):
        for node1 in vm1.task_list:
            for node2 in vm2.task_list:
                if self.G.has_edge(node1, node2) or self.G.has_edge(node2, node1):
                    return True
        return False

    def generate_vm_string(self, i, vm):
        s = "vm_type: " + str(vm.vm_type) + "\n"
        s = s + "vm: " + str(i) + "\ntask: "
        for task in vm.task_list:
            s = s + str(task) + ","
        s = s + "\nstart_time: " + str(vm.vm_start)
        s = s + "\nend_time: " + str(vm.vm_end)
        return s

    def dumpJSON(self):
        start = 0
        end = self.vertex_num - 1
        print("{")
        print("  \"nodes\": [")
        for u in range(start, end):
            print("        { \"order\":", str(self.G.node[u]["order"]) + ",")
            print("          \"name\":", "\"" + self.G.node[u]["name"] + "\",")
            print("          \"EST\":", str(self.G.node[u]["est"]) + ",")
            print("          \"EFT\":", str(self.G.node[u]["eft"]) + ",")
            print("          \"LST\":", str(self.G.node[u]["lst"]) + ",")
            print("          \"LFT\":", str(self.G.node[u]["lft"]))
            print("        },")

        print("        { \"order\":", str(self.G.node[end]["order"]) + ",")
        print("          \"name\":", "\"" + self.G.node[end]["name"] + "\",")
        print("          \"EST\":", str(self.G.node[end]["est"]) + ",")
        print("          \"EFT\":", str(self.G.node[end]["eft"]) + ",")
        print("          \"LST\":", str(self.G.node[end]["lst"]) + ",")
        print("          \"LFT\":", str(self.G.node[end]["lft"]))
        print("        }")
        print("  ],")
        print("  \"links\": [")

        num_edges = self.G.number_of_edges()
        nedge = 0
        for (u, v) in self.G.edges():
            nedge += 1
            print("        { \"source\":", "\"" + self.G.node[u]["name"] + "\",")
            print("          \"target\":", "\"" + self.G.node[v]["name"] + "\",")
            print("          \"throughput\":", str(self.G[u][v]["weight"]))
            if nedge < num_edges:
                print("        },")
            else:
                print("        }")

        print("    ]")
        print("}")

    def printGraphTimes(self):

        trow = "\nname     "
        for n in range(0, self.vertex_num):
            trow += self.G.node[n]["name"]
            trow += "  "
        print(trow)

        trow = "VM       "
        for n in range(0, self.vertex_num):
            trow += str(self.assigned_list[n])
            trow += "  "
        print(trow)

        trow = "perf     "
        for n in range(0, self.vertex_num):
            vm = self.assigned_list[n]
            if vm < 0:
                vm = 0
            trow += str(self.p_table[vm, n])
            trow += "  "
        print(trow)

        trow = "\nEST      "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["est"])
            trow += "  "
        print(trow)

        trow = "EFT      "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["eft"])
            trow += "  "
        print(trow)

        trow = "LST      "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["lst"])
            trow += "  "
        print(trow)

        trow = "LFT      "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["lft"])
            trow += "  "
        print(trow + "\n")

        trow = "EFT-EST  "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["eft"] - self.G.node[n]["est"])
            trow += "  "
        print(trow)

        trow = "LFT-LST  "
        for n in range(0, self.vertex_num):
            trow += str(self.G.node[n]["lft"] - self.G.node[n]["lst"])
            trow += "  "
        print(trow + "\n")

    def getStartCost(self):
        cost = 0
        for n in range(0, self.vertex_num):
            cost += self.vm_price[0] * (self.G.node[n]["eft"] - self.G.node[n]["est"])
        return (cost, self.G.node[self.vertex_num - 1]["eft"])

    def print_instances(self, tot_idle):
        total_cost = 0
        nS3 = 0
        nS2 = 0
        nS1 = 0
        print("\nPCP solution for task graph with " + str(self.vertex_num) + " nodes")
        rstr = ""
        if len(self.instances) > 0:
            rstr += "\n        Start time    Stop time    Duration    Inst cost    Assigned tasks"
            nodes_in_inst = 0
            for c in self.instances:
                task_list = sorted(c.task_list)
                nodes_in_inst += len(task_list)
                serv = c.vm_type
                if serv == 0:
                    nS1 += 1
                    ninst = nS1
                elif serv == 1:
                    nS2 += 1
                    ninst = nS2
                elif serv == 2:
                    nS3 += 1
                    ninst = nS3
                est = self.G.node[task_list[0]]["est"]
                eft = self.G.node[task_list[len(task_list) - 1]]["eft"]
                duration = eft - est
                rstr += "\nS" + str(serv) + "," + str(ninst)
                rstr += "      " + str(est) + "         " + str(eft) + "        " + str(duration)

                cost = duration * self.vm_price[serv]
                total_cost += cost
                rstr += "            " + str(cost)
                pcp_str = ""
                for j in task_list:
                    pcp_str += " " + self.G.node[j]["name"]
                rstr += "       " + pcp_str
        print(rstr)

        tot_non_inst = 0
        extra_cost = 0
        print("\ntotal instance cost: " + str(total_cost))
        if (nodes_in_inst != self.vertex_num):
            nonp = self.getNonInstanceNodes()
            nonstr = ""
            # print nonp
            for j in range(0, self.vertex_num):
                if nonp[j] == 0:
                    nonstr += "," + self.G.node[j]["name"]
                    extra_cost += (self.G.node[j]["eft"] - self.G.node[j]["est"]) * self.vm_price[0]
                    tot_non_inst += 1
            print("\n" + str(tot_non_inst) + " non instance nodes: " + nonstr[1:] + " with extra cost: " + str(
                extra_cost))
            total_cost += extra_cost
        if tot_idle > 0:
            print("\nTotal cost for " + str(self.vertex_num) + " nodes: " + str(total_cost) + " with tot idle=" + str(
                tot_idle))
        else:
            print("\nTotal cost for " + str(self.vertex_num) + " nodes: " + str(total_cost))
        return

    def getNonInstanceNodes(self):
        nonp = []
        for j in range(0, self.vertex_num):
            nonp.append(0)
        for c in self.instances:
            task_list = sorted(c.task_list)
            if len(task_list) > 0:
                for j in task_list:
                    nonp[j] = 1
        return nonp

    def updateGraphTimes(self):
        self.graphAssignEST(self.vertex_num - 1)
        self.graphAssignLFT(0)

    def graphAssignEST(self, d):
        self.visited = []
        for i in range(0, self.vertex_num):
            self.visited.append(0)
        self.graphCalcEFT(d)

    def graphCalcEFT(self, d):
        if self.visited[d] == 1:
            return self.G.node[d]["eft"]

        self.visited[d] = 1

        (ninstance, nservice) = self.getInstanceAndService(d)

        maxest = 0

        predecessors = []
        c_iter = iter(self.G.predecessors(d))
        while True:
            try:
                p = next(c_iter)
                predecessors.append(p)
            except StopIteration:
                break

        for p in predecessors:
            (pinstance, pservice) = self.getInstanceAndService(p)

            est = self.graphCalcEFT(p)

            lcost = self.G[p][d]["weight"]

            if pservice == nservice:
                if pservice == -1 or nservice == -1:
                    est += lcost
                elif pinstance == -1 or ninstance == -1 or pinstance != ninstance:
                    est += lcost
            else:
                est += lcost
            if est > maxest:
                maxest = est

        # node with no parents has zero EST
        ceft = maxest

        self.G.node[d]["est"] = ceft
        if nservice == -1:
            ceft += self.p_table[0, d]
        elif nservice == 0:
            ceft += self.p_table[0, d]
        elif nservice == 1:
            ceft += self.p_table[1, d]
        elif nservice == 2:
            ceft += self.p_table[2, d]
        else:
            ceft += self.p_table[0, d]

        self.G.node[d]["eft"] = ceft

        # if verbose>1:
        #    print G.node[d]["name"]+": EST="+str(G.node[d]["EST"]),",","EFT="+str(G.node[d]["EFT"])

        return self.G.node[d]["eft"]

    def graphAssignLFT(self, d):
        self.visited = []
        for i in range(0, self.vertex_num):
            self.visited.append(0)
        self.graphCalcLST(d)

    def graphCalcLST(self, d):
        if self.visited[d] == 1:
            return self.G.node[d]["lst"]

        self.visited[d] = 1
        (ninstance, nservice) = self.getInstanceAndService(d)

        minlft = self.deadline

        successors = []
        c_iter = iter(self.G.successors(d))
        while True:
            try:
                c = next(c_iter)
                successors.append(c)
            except StopIteration:
                break

        for c in successors:

            (cinstance, cservice) = self.getInstanceAndService(c)
            lft = self.graphCalcLST(c)

            lcost = self.G[d][c]["weight"]

            if cservice == nservice:
                if cservice == -1 or nservice == -1:
                    lft -= lcost
                elif cinstance == -1 or ninstance == -1 or cinstance != ninstance:
                    lft -= lcost
            else:
                lft -= lcost

            if lft < minlft:
                minlft = lft

        # node with no children has LFT equals deadline
        clft = minlft
        self.G.node[d]["lft"] = clft
        if nservice == -1:
            clft -= self.p_table[0, d]
        elif nservice == 0:
            clft -= self.p_table[0, d]
        elif nservice == 1:
            clft -= self.p_table[1, d]
        elif nservice == 2:
            clft -= self.p_table[2, d]
        else:
            clft -= self.p_table[0, d]

        self.G.node[d]["lst"] = clft

        return self.G.node[d]["lst"]

    def checkGraphTimes(self):

        retVal1 = self.graphCheckEST()
        retVal2 = self.graphCheckLFT()
        if retVal1 < 0 or retVal2 < 0:
            return -1
        else:
            return 0

    def graphCheckEST(self):
        for n in range(0, self.vertex_num):

            (ninstance, nservice) = self.getInstanceAndService(n)

            maxest = 0
            dominant_parent = -1
            p_iter = iter(self.G.predecessors(n))
            while True:
                try:
                    p = next(p_iter)
                    (pinstance, pservice) = self.getInstanceAndService(p)
                    est = self.G.node[p]["eft"]
                    lcost = self.G[p][n]["weight"]

                    if pservice == nservice:
                        # if pservice == 0 or nservice == 0 :
                        #    est += lcost
                        if pinstance == -1 or ninstance == -1 or pinstance != ninstance:
                            est += lcost
                    else:
                        est += lcost
                    if est > maxest:
                        maxest = est
                        dominant_parent = p
                except StopIteration:
                    break

            # node with no parents has zero EST
            if maxest > self.deadline:
                print("\n**** Wrong EST max: " + "EST(" + self.G.node[n]["name"] + ")=" + str(
                    self.G.node[n]["est"]) + " and " + "EST(" + self.G.node[dominant_parent]["name"] + ")=" + str(
                    maxest))
                return -1
            elif self.G.node[n]["est"] < maxest:
                print("\n**** EST mismatch: " + "EST(" + self.G.node[n]["name"] + ")=" + str(
                    self.G.node[n]["est"]) + " < " + "EST(" + self.G.node[dominant_parent]["name"] + ")=" + str(maxest))
                return -1

        return 0

    def graphCheckLFT(self):
        for n in range(0, self.vertex_num):

            (ninstance, nservice) = self.getInstanceAndService(n)

            minlft = self.deadline
            dominant_child = -1
            c_iter = iter(self.G.successors(n))
            while True:
                try:
                    c = next(c_iter)
                    (cinstance, cservice) = self.getInstanceAndService(c)

                    lft = self.G.node[c]["lst"]
                    lcost = self.G[n][c]["weight"]

                    if cservice == nservice:
                        # if cservice == 0 or nservice == 0 :
                        #    lft -= lcost
                        if cinstance == -1 or ninstance == -1 or cinstance != ninstance:
                            lft -= lcost
                    else:
                        lft -= lcost

                    if lft < minlft:
                        dominant_child = c
                        minlft = lft

                except StopIteration:
                    break

            # node with no children has LFT equals deadline
            if minlft < 0:
                print("\n**** Negative LFT : " + "LFT(" + self.G.node[n]["name"] + ")=" + str(
                    self.G.node[n]["lft"]) + " and " + "LFT(" + self.G.node[dominant_child]["name"] + ")=" + str(
                    minlft))
                return -1
            elif self.G.node[n]["lft"] > minlft:
                print("\n**** LFT mismatch: " + "LFT(" + self.G.node[n]["name"] + ")=" + str(
                    self.G.node[n]["lft"]) + " > " + "LFT(" + self.G.node[dominant_child]["name"] + ")=" + str(minlft))
                return -1

        return 0

    def checkIdleTime(self):
        tot_idle = 0
        idles = "\n"
        for i in range(0, len(self.instances)):
            inst = self.instances[i]
            task_list = sorted(inst.task_list)
            if len(task_list) > 1:
                for j in range(0, len(task_list) - 1):
                    idle_time = self.G.node[task_list[j + 1]]["est"] - self.G.node[task_list[j]]["eft"]
                    if idle_time > 0:
                        tot_idle += idle_time
                        idles += "\n Instance[" + str(i) + "] constains idle time: " + "EST(" + \
                                 self.G.node[task_list[j + 1]]["name"] + ")-EFT(" + self.G.node[task_list[j]][
                                     "name"] + ")>0"
        print(idles)
        return tot_idle

    def getInstanceAndService(self, d):
        if len(self.instances) > 0:
            for i in range(0, len(self.instances)):
                inst = self.instances[i]
                task_list = sorted(inst.task_list)
                if d in task_list:
                    return (i, inst.vm_type)
            return (-1, -1)
        else:
            return (-1, -1)

    def update_node(self, n):
        (ninstance, nservice) = self.getInstanceAndService(n)

        maxest = 0
        for parent in self.G.predecessors(n):
            (pinstance, pservice) = self.getInstanceAndService(parent)
            est = self.G.node[parent]["eft"]
            lcost = self.G[parent][n]["weight"]

            if pservice == nservice:

                if pinstance == -1 or ninstance == -1 or pinstance != ninstance:
                    est += lcost
            else:
                est += lcost
            if est > maxest:
                maxest = est

        self.G.node[n]['est'] = maxest
        if self.assigned_list[n] == -1:
            self.G.node[n]['eft'] = maxest + self.p_table[0, n]
        else:
            self.G.node[n]['eft'] = maxest + self.p_table[self.assigned_list[n], n]

        minlft = self.deadline
        for child in self.G.successors(n):
            (cinstance, cservice) = self.getInstanceAndService(child)
            lft = self.G.node[child]["lst"]
            lcost = self.G[n][child]["weight"]

            if cservice == nservice:

                if cinstance == -1 or ninstance == -1 or cinstance != ninstance:
                    lft -= lcost
            else:
                lft -= lcost
            if lft < minlft:
                minlft = lft

        self.G.node[n]['lft'] = minlft
        if self.assigned_list[n] == -1:
            self.G.node[n]['lst'] = minlft - self.p_table[0, n]
        else:
            self.G.node[n]['lst'] = minlft - self.p_table[self.assigned_list[n], n]


if __name__ == '__main__':
    print(os.getcwd())
    wf = Workflow()
    workflow_file = ''
    performance_file = ''
    price_file = ''
    deadline_file = ''
    infrastructure_file = ''
    SDI_file = ''

    usage = "usage: %prog options name"
    parser = OptionParser(usage)
    parser.set_defaults(runs=1, iters=1)
    parser.add_option("-d", "--dir", dest="dir", help="specify input directory", type="string", default="input")
    parser.add_option("-i", "--file", dest="file", help="specify input file", type="string", default="")
    parser.add_option("-j", "--jason", dest="json", help="dump json file", type="int", default="0")
    parser.add_option("-p", "--perc", dest="perc", help="cp percentage deadline", type="int", default="-1")
    # parser.add_option("-v", "--verbose",  dest="verbose", help="verbose output", type="int", default="0")
    (options, args) = parser.parse_args()

    if options.file:
        workflow_file = options.dir + '/' + options.file + '/' + options.file + '.dag'
        performance_file = options.dir + '/' + options.file + '/performance'
        deadline_file = options.dir + '/' + options.file + '/deadline'
        price_file = options.dir + '/' + options.file + '/price'
        infrastructure_file = options.dir + '/' + options.file + '/inf'
    else:
        sys.exit("\nERROR - Missing option -f or --file.\n")

    print("Networkx" + nx.__version__)

    '../input/workflow', '../input/performance.txt', '../input/price.txt', '../input/Deadline'
    wf.init(workflow_file, performance_file, price_file, deadline_file)
    wf.calc_startConfiguration(options.perc)

    start_cost, start_eft = wf.getStartCost()
    start_str = "start configuartion: cost=" + str(start_cost) + "  EFT(exit)=" + str(start_eft)
    print("\nStart situation")
    wf.printGraphTimes()

    wf.ic_pcp()
    print("\nEnd situation")
    wf.printGraphTimes()

    # entry and exit node not part of PCP, so
    # adjust LST, LFT of entry node
    # adjust EST, EFT of exit node
    wf.update_node(0)
    wf.update_node(wf.number_of_nodes() - 1)

    # check PCP end situation
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
