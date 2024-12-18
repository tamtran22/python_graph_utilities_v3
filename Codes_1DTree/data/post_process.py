import numpy as np
import os

def print_tecplot_file(
        nodes={
            'x': np.array([1.,2,3,4]),
            'y': np.array([6.,7,8,9]),
            'z': np.array([1,2,4,9.2])
        },
        elements=np.array([[1,2],[2,3], [2,4]]), 
        ZONE_T='output.dat', 
        F='FEPoint', 
        ET='LINESEG',
        file_name='output.dat'
    ):
    # check input
    try:
        n_field = len(list(nodes)) # number of fields
        for field in nodes:
            if len(np.shape(nodes[field])) != 1:
                raise KeyError
        n_node = np.shape(nodes[list(nodes.keys())[0]])[0] # number of nodes
        for i in range(1, n_field):
            if n_node != np.shape(nodes[list(nodes.keys())[i]])[0]:
                raise KeyError
        if len(np.shape(elements)) != 2:
            raise KeyError
        n_element = np.shape(elements)[0] # number of elements
        n_node_per_element = np.shape(elements)[1]
    except:
        raise KeyError

    # start
    f = open(file_name, 'w+')
    # header
    f.write('VARIABLES=')
    for field in nodes:
        f.write(f'"{field}" ')
    f.write('\n')
    f.write(f'ZONE T= "{ZONE_T}"\n')
    f.write(f'N={n_node}, E={n_element}, F={F}, ET={ET}\n')
    # data
    for i in range(n_node):
        for field in nodes:
            f.write(f' {nodes[field][i]}')
        f.write('\n')
    
    for i in range(n_element):
        for j in range(n_node_per_element):
            f.write(f' {elements[i][j]}')
        f.write('\n')
    # finish
    f.close()

def print_1D(edge_index, coordinate, node_attr=(None ,None), dir='./test'):
    os.system(f'mkdir {dir}')
    if node_attr[1] is None:
        n_time = 1
    else:
        n_time = node_attr.shape[1]
    for i in range(n_time):
        print_tecplot_file(
            nodes={
                'x': coordinate[:,0],
                'y': coordinate[:,1],
                'z': coordinate[:,2],
                # 'attr': node_attr[:,i]
            },
            elements=np.transpose(edge_index) + 1,
            ZONE_T=f'plt_nd_{str(i).zfill(6)}.dat',
            file_name=dir+f'/plt_nd_{str(i).zfill(6)}.dat'
        )