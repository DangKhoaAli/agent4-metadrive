"""Dependency-free unit tests for action calibration math."""
from calibrate_metadrive_v21 import angle_delta, summarize_steering, summarize_longitudinal
from audit_pdd_v21 import corr, pct_sign_agree
from math import pi

def main():
    assert abs(angle_delta(-pi+0.1, pi-0.1)-0.2)<1e-8
    assert abs(angle_delta(pi-0.1,-pi+0.1)+0.2)<1e-8
    rows=[]
    for seed in (0,1):
        for steer,yaw,lat in ((0.,.01,0.),(.25,.12,.25),(-.25,-.12,-.25),(.4,.20,.4),(-.4,-.20,-.4)):
            rows.append({'seed':seed,'test_action':[steer,.35], 'yaw_delta_rad':yaw,
                         'left_displacement_m':lat,'early_end':False,
                         'pulse_steps_completed':7,'pulse_steps_requested':7})
    steer=summarize_steering(rows)
    assert steer['bidirectional_sign_supported']
    assert steer['candidate_carla_to_metadrive_steer_sign_not_applied']==-1
    assert steer['physical_direction_of_positive_metadrive_steer']=='left'
    act=[]
    for seed in (0,1):
        for a2, dv in ((0.,0.05),(.2,.25),(.4,.5)):
            act.append({'seed':seed,'test_action':[0.,a2],
                        'speed_delta_mps':dv,'early_end':False,
                        'pulse_steps_completed':12,'pulse_steps_requested':12})
    summ=summarize_longitudinal(act,'per_seed')
    assert summ['neutral_corrected_medians_mps']['0.4']>0.4
    assert corr([(x,2*x+1) for x in range(40)])>0.99
    assert pct_sign_agree([(x,x) for x in range(10)],1,1)['same_sign_pct']==100.
    print('Calibration helper tests: PASS')

if __name__=='__main__': main()
