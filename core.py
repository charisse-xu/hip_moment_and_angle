import os
import sys
third_party_path = os.path.abspath(os.path.join(__file__, '../third_party'))
sys.path.insert(0, third_party_path)
from sage.base_app import BaseApp
import sklearn

if __name__ == '__main__':
    from gait_phase import GaitPhase
    # from knee_moments import MomentPrediction
    from knee_moments_ import MomentPrediction
    from const import WAIT_PREDICTION, PREDICTION_DONE, R_FOOT
else:
    from .gait_phase import GaitPhase
    # from .knee_moments import MomentPrediction
    from .knee_moments_ import MomentPrediction
    from .const import WAIT_PREDICTION, PREDICTION_DONE, R_FOOT
sys.path.remove(third_party_path)

class Core(BaseApp):
    def __init__(self, my_sage):
        BaseApp.__init__(self, my_sage, __file__)
        self.gait_phase = GaitPhase()
        self.moment_prediction = MomentPrediction(self.config['weight'], self.config['height'])
        self.time_now = 0

    def run_in_loop(self):
        data = self.my_sage.get_next_data()
        self.gait_phase.update_gaitphase(data[R_FOOT])
        data_to_send = self.moment_prediction.update_stream(data, self.gait_phase)
        for data, hfm, ham, hfa, haa, stance_flag in data_to_send:
            self.time_now += 0.01
            my_data = {'time': [self.time_now], 'HFM': [hfm], 'HAM': [ham], \
                       'HFA': [hfa], 'HAA': [haa], 'Stance_Flag': [stance_flag]}
            self.my_sage.send_stream_data(data, my_data)
            self.my_sage.save_data(data, my_data)
        return True

if __name__ == '__main__':
    # This is only for testing. make sure you do the pairing first in web api
    from sage.sage import Sage
    app = Core(Sage())
    app.test_run()
