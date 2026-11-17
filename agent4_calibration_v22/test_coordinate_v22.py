from coordinate_patch_v22 import determine_local_y_to_right_multiplier, correct_features_to_pdd_right, carla_to_metadrive_steer

class LeftPositiveAgent:
    heading_theta = 0.0
    def convert_to_local_coordinates(self, delta, _):
        return delta

class RightPositiveAgent:
    heading_theta = 0.0
    def convert_to_local_coordinates(self, delta, _):
        return (delta[0], -delta[1])

s, info = determine_local_y_to_right_multiplier(LeftPositiveAgent())
assert s == -1 and info['local_y_positive'] == 'left'
s, info = determine_local_y_to_right_multiplier(RightPositiveAgent())
assert s == +1 and info['local_y_positive'] == 'right'
features = {
    'wp1_right_m':-3.5, 'wp2_right_m':-1.2,
    'left_vehicle_dist_m':3.0, 'right_vehicle_dist_m':20.0,
    'has_left_vehicle':1.0, 'has_right_vehicle':0.0,
}
patched = correct_features_to_pdd_right(features, -1)
assert patched['wp1_right_m']==3.5
assert patched['wp2_right_m']==1.2
assert patched['left_vehicle_dist_m']==20.0
assert patched['right_vehicle_dist_m']==3.0
assert patched['has_right_vehicle']==1.0
assert features['wp1_right_m']==-3.5  # never mutate input
assert carla_to_metadrive_steer(+.25) == -.25
assert carla_to_metadrive_steer(-.25) == +.25
print('V2.2 coordinate/sign helper tests: PASS')
