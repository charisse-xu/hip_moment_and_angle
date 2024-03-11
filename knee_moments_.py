import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import numpy as np
from collections import deque
from .const import WAIT_PREDICTION, IMU_LIST, IMU_FIELDS, PREDICTION_DONE, MAX_BUFFER_LEN,\
    R_FOOT, WEIGHT_LOC, HEIGHT_LOC, ACC_ALL, GYR_ALL, GRAVITY
import os

LSTM_UNITS, FCNN_UNITS = 40, 40
device = 'cpu'
class InertialNet(nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        # 看下这个怎么改
        self.rnn_layer = nn.LSTM(x_dim, LSTM_UNITS, nlayer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        for name, param in self.rnn_layer.named_parameters():
            # print(name, param.data, param.data.shape) 160x24
            if 'weight' in name:
                nn.init.xavier_normal_(param) # 再初始化一下？

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        # 进入RNN需要压缩再解压
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence) # 单层?
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class VideoNet(InertialNet):
    pass


class OutNet(nn.Module):
    def __init__(self, input_dim, output_dim, high_level_locs=[2, 3, 4]):
        super(OutNet, self).__init__()
        self.high_level_locs = high_level_locs
        self.linear_1 = nn.Linear(input_dim + len(high_level_locs), FCNN_UNITS, bias=True)
        self.linear_2 = nn.Linear(FCNN_UNITS, output_dim, bias=True)
        self.relu = nn.ReLU()
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        if len(self.high_level_locs) > 0:
            sequence = torch.cat((sequence, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        height = others[:, 0, HEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence


class LmfImuOnlyNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, acc_dim, gyr_dim, output_dim=2):
        super(LmfImuOnlyNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.vid_subnet = VideoNet(24, 'vid net', seed=0) # 被pass了
        self.rank = 10
        self.fused_dim = 40

        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*LSTM_UNITS + 1, self.fused_dim))
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*LSTM_UNITS + 1, self.fused_dim))
        self.vid_factor = Parameter(torch.Tensor(self.rank, 1, 2*LSTM_UNITS + 1, self.fused_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

        # init factors
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.vid_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        # if acc_dim <= 3:
        self.out_net = OutNet(self.fused_dim, output_dim, [])  # do not use high level features
        # else:
            # self.out_net = OutNet(self.fused_dim, [2])  # only use FPA from high level features

    def __str__(self):
        return 'LMF IMU only net'

    def set_scalers(self, scalers):
        self.scalers = scalers

    def forward(self, acc_x, gyr_x, vid_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        # data_type = torch.cuda.FloatTensor
        data_type = torch.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type).to(device), requires_grad=False), acc_h), dim=2)
        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type).to(device), requires_grad=False), gyr_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_vid = torch.full_like(fusion_acc, 1) # 这什么意思
        fusion_zy = fusion_acc * fusion_gyr * fusion_vid
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
        sequence = self.out_net(sequence, others)
        return sequence

# class InertialNet(torch.nn.Module):
#     def __init__(self, x_dim, net_name, seed=0, nlayer=1):
#         super(InertialNet, self).__init__()
#         self.net_name = net_name
#         torch.manual_seed(seed)
#         self.rnn_layer = torch.nn.LSTM(x_dim, LSTM_UNITS, nlayer, batch_first=True, bidirectional=True)
#         self.dropout = torch.nn.Dropout(0.2)
#         for name, param in self.rnn_layer.named_parameters():
#             if 'weight' in name:
#                 torch.nn.init.xavier_normal_(param)

#     def __str__(self):
#         return self.net_name

#     def forward(self, sequence, lens):
#         sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
#         sequence, _ = self.rnn_layer(sequence)
#         sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=152)
#         sequence = self.dropout(sequence)
#         return sequence

# class OutNet(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(OutNet, self).__init__()
#         self.linear_1 = torch.nn.Linear(input_dim, FCNN_UNITS, bias=True)
#         self.linear_2 = torch.nn.Linear(FCNN_UNITS, 2, bias=True)
#         self.relu = torch.nn.ReLU()
#         for layer in [self.linear_1, self.linear_2]:
#             torch.nn.init.xavier_normal_(layer.weight)

#     def forward(self, sequence, others):
#         sequence = self.linear_1(sequence)
#         sequence = self.relu(sequence)
#         sequence = self.linear_2(sequence)
#         weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
#         height = others[:, 0, HEIGHT_LOC].unsqueeze(1).unsqueeze(2)
#         sequence = torch.div(sequence, weight * GRAVITY * height / 100)
#         return sequence


# class VideoNet(InertialNet):
#     pass


# class LmfNet(nn.Module):
#     """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
#     def __init__(self, acc_dim, gyr_dim):
#         super(LmfNet, self).__init__()
#         self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
#         self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
#         self.vid_subnet = VideoNet(24, 'vid net', seed=0)
#         self.rank = 10
#         self.fused_dim = 40

#         self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
#         self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
#         self.vid_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
#         self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
#         self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

#         # init factors
#         nn.init.xavier_normal_(self.acc_factor, 10)
#         nn.init.xavier_normal_(self.gyr_factor, 10)
#         nn.init.xavier_normal_(self.vid_factor, 10)
#         nn.init.xavier_normal_(self.fusion_weights)
#         self.fusion_bias.data.fill_(0)

#         self.out_net = OutNet(self.fused_dim)

#     def __str__(self):
#         return 'LMF fusion net'

#     def set_scalers(self, scalers):
#         self.scalers = scalers

#     def forward(self, acc_x, gyr_x, vid_x, others, lens):
#         acc_h = self.acc_subnet(acc_x, lens)
#         gyr_h = self.gyr_subnet(gyr_x, lens)
#         vid_h = self.vid_subnet(vid_x, lens)
#         batch_size = acc_h.data.shape[0]
#         data_type = torch.FloatTensor

#         _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type), requires_grad=False), acc_h), dim=2)
#         _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type), requires_grad=False), gyr_h), dim=2)
#         _vid_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, vid_x.shape[1], 1).type(data_type), requires_grad=False), vid_h), dim=2)

#         fusion_acc = torch.matmul(_acc_h, self.acc_factor)
#         fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
#         fusion_vid = torch.matmul(_vid_h, self.vid_factor)
#         fusion_zy = fusion_acc * fusion_gyr * fusion_vid

#         # permute to make batch first
#         sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
#         sequence = self.out_net(sequence, others)
#         return sequence


# class LmfImuOnlyNet(LmfNet):
#     """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
#     def __init__(self, acc_dim, gyr_dim):
#         super(LmfImuOnlyNet, self).__init__(acc_dim, gyr_dim)
#         if acc_dim <= 3:
#             self.out_net = OutNet(self.fused_dim)  # do not use high level features
#         else:
#             self.out_net = OutNet(self.fused_dim)  # only use FPA from high level features

#     def __str__(self):
#         return 'LMF IMU only net'

#     def forward(self, acc_x, gyr_x, vid_x, others, lens):
#         acc_h = self.acc_subnet(acc_x, lens)
#         gyr_h = self.gyr_subnet(gyr_x, lens)
#         batch_size = acc_h.data.shape[0]
#         data_type = torch.FloatTensor

#         _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type), requires_grad=False), acc_h), dim=2)
#         _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type), requires_grad=False), gyr_h), dim=2)

#         fusion_acc = torch.matmul(_acc_h, self.acc_factor)
#         fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
#         fusion_vid = torch.full_like(fusion_acc, 1)
#         fusion_zy = fusion_acc * fusion_gyr * fusion_vid
#         # permute to make batch first
#         sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
#         sequence = self.out_net(sequence, others)
#         return sequence

class MomentPrediction:
    def __init__(self, weight, height):
        self.data_buffer = deque(maxlen=MAX_BUFFER_LEN)
        self.data_margin_before_step = 20
        self.data_margin_after_step = 20
        self.data_array_fields = [axis + '_' + sensor for sensor in IMU_LIST for axis in IMU_FIELDS]

        base_path = os.path.abspath(os.path.dirname(__file__))
        model_path = base_path + '/models/8IMU_LSTM40_FCNN40_4OUT.pth'
        self.model = torch.load(model_path)
        # self.model = LmfImuOnlyNet(24, 24) # 3*8
        # self.model.eval()
        # self.model.load_state_dict(torch.load(model_state_path))
        
        self.model.acc_col_loc = [self.data_array_fields.index(field) for field in ACC_ALL]
        self.model.gyr_col_loc = [self.data_array_fields.index(field) for field in GYR_ALL]
        self.model.vid_col_loc = [0]

        self.weight = weight
        self.height = height

        anthro_data = np.zeros([1, 152, 2], dtype=np.float32)
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        anthro_data[:, :, HEIGHT_LOC] = self.height
        self.model_inputs = {'others': torch.from_numpy(anthro_data), 'step_length': None,
                             'input_acc': None, 'input_gyr': None, 'input_vid': None}

    def update_stream(self, data, gait_phase):
        # data, hfm, ham, hfa, haa, stance_flag
        self.data_buffer.append([data, 0., 0., 0., 0., 0])
        package = data[R_FOOT]['Package']
        if gait_phase.current_phase == WAIT_PREDICTION:
            if package - gait_phase.off_package >= self.data_margin_after_step - 1:
                step_length = int(gait_phase.off_package - gait_phase.strike_package + self.data_margin_before_step + self.data_margin_after_step)
                if step_length <= len(self.data_buffer):
                    # start = time.time()
                    inputs = self.transform_input(step_length, self.data_buffer, self.model_inputs)
                    pred = self.model(inputs['input_acc'], inputs['input_gyr'], inputs['input_vid'], inputs['others'], inputs['step_length'])
                    pred = pred.detach().numpy().astype(np.float)[0]
                    for i_sample in range(step_length):
                        self.data_buffer[-step_length+i_sample][1:5] = [pred[i_sample, :]] ## 改动
                    for i_sample in range(self.data_margin_before_step, step_length - self.data_margin_after_step):
                        self.data_buffer[-step_length+i_sample][5] = 1
                    # duration = time.time() - start
                    # print(duration)

                gait_phase.current_phase = PREDICTION_DONE
        if len(self.data_buffer) == MAX_BUFFER_LEN:
            return [self.data_buffer.popleft()]
        return []

    def transform_input(self, step_length, data_buffer, model_inputs):
        raw_data = []
        for sample_data in list(data_buffer)[-step_length:]:
            raw_data_one_row = []
            for i_sensor in range(len(IMU_LIST)):
                raw_data_one_row.extend([sample_data[0][i_sensor][field] for field in IMU_FIELDS])
            raw_data.append(raw_data_one_row)
        data = np.array(raw_data, dtype=np.float32)
        data[:, self.model.acc_col_loc] = self.normalize_array_separately(
            data[:, self.model.acc_col_loc], self.model.scalers['input_acc'], 'transform')
        model_inputs['input_acc'] = torch.from_numpy(np.expand_dims(data[:, self.model.acc_col_loc], axis=0))
        data[:, self.model.gyr_col_loc] = self.normalize_array_separately(
            data[:, self.model.gyr_col_loc], self.model.scalers['input_gyr'], 'transform')
        model_inputs['input_gyr'] = torch.from_numpy(np.expand_dims(data[:, self.model.gyr_col_loc], axis=0))
        data[:, :, self.model.vid_col_loc] = self.normalize_array_separately(
            data[:, :, self.model.vid_col_loc], self.model.scalers['input_vid'], 'transform')
        model_inputs['input_vid'] = torch.from_numpy(np.expand_dims(data[:, self.model.vid_col_loc], axis=0))

        model_inputs['step_length'] = torch.tensor([step_length], dtype=torch.int32)
        return model_inputs

    @staticmethod
    def normalize_array_separately(data, scaler, method, scaler_mode='by_each_column'):
        input_data = data.copy()
        original_shape = input_data.shape
        target_shape = [-1, input_data.shape[2]] if scaler_mode == 'by_each_column' else [-1, 1]
        input_data[(input_data == 0.).all(axis=2), :] = np.nan
        input_data = input_data.reshape(target_shape)
        scaled_data = getattr(scaler, method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        scaled_data[np.isnan(scaled_data)] = 0.
        return scaled_data



