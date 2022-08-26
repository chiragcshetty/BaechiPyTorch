## NOTE: Node ids must start with 0 for m_sct solver to work

# example in the paper
graph_fig1 = {}
graph_fig1[0] = {
    'id'             : 0,
    'forward_time'   : 1.0,
    'permanent_mem'  : 1.0,
    'temporary_mem'  : 0.0,
    'children'       : [1, 4],
    'edge_weight'    : [1.0, 1.0]
}
graph_fig1[1] = {
    'id'             : 1,
    'forward_time'   : 3.0,
    'permanent_mem'  : 2.0,
    'temporary_mem'  : 0.0,
    'children'       : [2],
    'edge_weight'    : [1.0]
}
graph_fig1[2] = {
    'id'             : 2,
    'forward_time'   : 3.0,
    'permanent_mem'  : 2.0,
    'temporary_mem'  : 0.0,
    'children'       : [3],
    'edge_weight'    : [1.0]
}
graph_fig1[3] = {
    'id'             : 3,
    'forward_time'   : 1.0,
    'permanent_mem'  : 1.0,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}
graph_fig1[4] = {
    'id'             : 4,
    'forward_time'   : 3.0,
    'permanent_mem'  : 1.0,
    'temporary_mem'  : 0.0,
    'children'       : [5],
    'edge_weight'    : [1.0]
}
graph_fig1[5] = {
    'id'             : 5,
    'forward_time'   : 1.0,
    'permanent_mem'  : 1.0,
    'temporary_mem'  : 0.0,
    'children'       : [3],
    'edge_weight'    : [1.0]
}


########################
# Case where keeping a device idle gives better makespan. but SCT does not does it
# It does same as ETF
graph_fig2 = {}

graph_fig2[0] = {
    'id'             : 0,
    'forward_time'   : 5.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [2],
    'edge_weight'    : [1.0]
}

graph_fig2[1] = {
    'id'             : 1,
    'forward_time'   : 4.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [2, 3],
    'edge_weight'    : [3.0, 1.0 ]
}

graph_fig2[2] = {
    'id'             : 2,
    'forward_time'   : 7.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

graph_fig2[3] = {
    'id'             : 3,
    'forward_time'   : 4.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

########################
# Case that satisfies FS1 in hanen paper
graph_fig21 = {}

graph_fig21[0] = {
    'id'             : 0,
    'forward_time'   : 9.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [2],
    'edge_weight'    : [2.0]
}

graph_fig21[1] = {
    'id'             : 1,
    'forward_time'   : 9.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [2, 3],
    'edge_weight'    : [3.0, 1.0 ]
}

graph_fig21[2] = {
    'id'             : 2,
    'forward_time'   : 8.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

graph_fig21[3] = {
    'id'             : 3,
    'forward_time'   : 5.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}


graph_fig21[4] = {
    'id'             : 4,
    'forward_time'   : 6.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [5],
    'edge_weight'    : [3.0]
}

graph_fig21[5] = {
    'id'             : 5,
    'forward_time'   : 4.0,
    'permanent_mem'  : 0.1,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

########################
## Example in Hanen SCT paper. 
# But SCT, ETF give same placement 

graph_fig3 = {}

graph_fig3[0] = {
    'id'             : 0 ,
    'forward_time'   : 6.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [2, 3, 4],
    'edge_weight'    : [1.0, 5.0, 2.0]
}

graph_fig3[1] = {
    'id'             : 1,
    'forward_time'   : 7.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [2, 3, 4],
    'edge_weight'    : [1.0, 1.0, 1.0]
}

graph_fig3[2] = {
    'id'             : 2,
    'forward_time'   : 9.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [5, 6],
    'edge_weight'    : [5.0, 3.0]
}

graph_fig3[3] = {
    'id'             : 3,
    'forward_time'   : 8.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [5, 6, 7],
    'edge_weight'    : [4.0, 1.0, 5.0]
}

graph_fig3[4] = {
    'id'             : 4,
    'forward_time'   : 10.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [8],
    'edge_weight'    : [1.0]
}

graph_fig3[5] = {
    'id'             : 5,
    'forward_time'   : 6.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

graph_fig3[6] = {
    'id'             : 6,
    'forward_time'   : 6.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

graph_fig3[7] = {
    'id'             : 7,
    'forward_time'   : 10.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

graph_fig3[8] = {
    'id'             : 8,
    'forward_time'   : 6.0,
    'permanent_mem'  : 0.01,
    'temporary_mem'  : 0.0,
    'children'       : [],
    'edge_weight'    : []
}

