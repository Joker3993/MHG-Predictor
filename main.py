from Subgraph import SubGraph
from Train import Tran
from Whole_graph import Whole_Graph

if __name__ == "__main__":
    list_eventlog = [
        # 'helpdesk',
        # 'bpi13_problems',
        # 'bpi13_closed_problems',
        'bpi12w_complete',
        # 'bpic2017_o',
        # 'bpi12_work_all',
    ]
    # 每个日志中的活动种类数量
    dict = {
        'helpdesk': 14,
        'bpi13_problems': 7,
        'bpi13_closed_problems': 4,
        'bpi12_all_complete': 23,
        'bpi12w_complete': 6,
        'bpic2017_o': 8,
        'bpi12_work_all': 19,
        'receipt': 27,
        'bpic2020': 19,
        'bpi13_incidents': 13,
    }

    for eventlog in list_eventlog:
        bin = dict[eventlog]
        get_Whole = Whole_Graph(eventlog, bin).whole_main()
        get_Subgraph = SubGraph(eventlog).Subgraph_main()
        Tran(eventlog).tran_main()
