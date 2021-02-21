from BSE2 import *


start_time = 0.0
end_time = 200.0
duration = end_time - start_time

range1 = (50 ,50)
range2 = (20 ,20)
supply_schedule = [{'from': 0, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]
# supply_schedule = [{'from': 0, 'to': 100, 'ranges': [(50,50)], 'stepmode': 'fixed'},
#                    {'from': 100, 'to': 200, 'ranges': [(50,150)], 'stepmode': 'fixed'},
#                    {'from': 200, 'to': 300, 'ranges': [(50,150)], 'stepmode': 'fixed'},
#                    {'from': 300, 'to': 500, 'ranges': [(50,50)], 'stepmode': 'fixed'}
#
#                    ]

range3 = (150, 150)
range4 = (100, 100)
demand_schedule = [{'from': 0, 'to': end_time, 'ranges': [range3], 'stepmode': 'fixed'}]
# demand_schedule = [{'from': 0, 'to': 100, 'ranges': [(150,150)], 'stepmode': 'fixed'},
#                    {'from': 100, 'to': 200, 'ranges': [(50,150)], 'stepmode': 'fixed'},
#                    {'from': 200, 'to': 300, 'ranges': [(150,150)], 'stepmode': 'fixed'},
#                    {'from': 300, 'to': 500, 'ranges': [(50,150)], 'stepmode': 'fixed'}
#
#                    ]

order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': 100,
               'timemode': 'periodic'}

## 'AAAA' holds the block order
buyers_spec = [('GDX',10),('AA',10),('ZIC',10),('ZIP',10)]
sellers_spec = [('GDX',10),('AA',10),('ZIC',10),('ZIP',10)]
# buyers_spec = [('BID_IGDX_3', 10), ('BID_IZIP_3', 10), ('BID_IAA_3', 10),('BID_ISHV_3', 10),  ('AAAA', 10)]
# sellers_spec = [('BID_IGDX_3', 10), ('BID_IZIP_3', 10), ('BID_IAA_3', 10),('BID_ISHV_3', 10),  ('AAAA', 10)]
traders_spec = {'sellers' :sellers_spec, 'buyers' :buyers_spec}

sys.stdout.flush()

fname = 'Mybalances.csv'
summary_data_file = open(fname, 'w')

fname = 'Mytapes.csv'
tape_data_file = open(fname, 'w')

fname = 'Myblotters.csv'
blotter_data_file = open(fname, 'w')

for session in range(100):
    sess_id = 'Test%02d' % session
    print('Session %s; ' % sess_id)


    market_session(sess_id, start_time, end_time, traders_spec, order_sched, summary_data_file, tape_data_file, blotter_data_file, True, False)

summary_data_file.close()
tape_data_file.close()
blotter_data_file.close()

print('\n Experiment Finished')
