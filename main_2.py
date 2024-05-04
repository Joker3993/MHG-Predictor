from DGLSubgraph import SubGraph
from DGLTrain_2 import Tran

"""没有三个时间特征的处理"""
from DGLWhole_graph import Whole_Graph

"""有三个时间特征的处理"""
# from 旧版代码.new_whole_graph import Whole_Graph

if __name__ == "__main__":
    list_eventlog = [
        # 'helpdesk',
        # 'bpi13_problems',
        'bpi13_closed_problems',
        # 'bpi12_all_complete',
        # 'bpi12w_complete',
        # 'bpic2017_o',
        # 'bpi12_work_all',
        # 'receipt',
        # 'bpic2020',
        # 'bpi13_incidents',
    ]
    #每个日志中的活动种类数量
    dict ={
        'helpdesk':14,
        'bpi13_problems':7,
        'bpi13_closed_problems':4,
        'bpi12_all_complete':23,
        'bpi12w_complete':6,
        'bpic2017_o':8,
        'bpi12_work_all':19,
        'receipt':27,
        'bpic2020':19,
        'bpi13_incidents':13,
    }

    for eventlog in list_eventlog:
        bin = dict[eventlog]
        get_Whole = Whole_Graph(eventlog, bin).whole_main()
        get_Subgraph = SubGraph(eventlog).Subgraph_main()
        Tran(eventlog).tran_main()


    """bpi13_problems: 7,bpic2017_o: 8 , receipt：27 , bpi13_incidents：13, 
    bpi12w_complete :6 ,bpi12_work_all: 19, bpi12_all_complete: 23 ,bpic2020: 19,
    helpdesk : 14 ,"""


