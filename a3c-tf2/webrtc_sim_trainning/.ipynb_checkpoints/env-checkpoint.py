import os, sys
import numpy as np
import math

TRACE_DIR = "../../dataset/cooked/"

class Env:
    
    # -- class method --
    def get_trace_files(trace_dir):
        """
        trace 파일을 리스트로 만들어서 반환함.
        
        Arguments:
        trace_dir - trace 파일이 있는 디렉토레 경로
        
        Return: trace 파일의 경로가 담긴 리스트
        """
        filepaths = []
        for filename in os.listdir(trace_dir):
            if filename[0] != ".":
                trace_path = trace_dir + filename
                filepaths.append(trace_path)
        return filepaths
    
    def __init__(self, random_seed=None, num_clu_low=1, num_clu_high=30, num_cli_low=1, num_cli_high=30):
        # -- random seed --
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        # diemension
        # rembs, number of clients, bandwidth, hardward, bitrate of source
        self.state_shape = (3, 30)
        self.action_shape = (30)
        
        # -- set variables --
        self.num_clu_low = num_clu_low
        self.num_clu_high = num_clu_high
        self.num_cli_low = num_cli_low
        self.num_cli_high = num_cli_high
        
        self.reset()
        
    def reset(self):
        # -- network configuration --
        # 클러스터 갯수를 랜덤으로 지정
        self.num_clu = np.random.randint(low=self.num_clu_low, high=self.num_clu_high)
        # 각 클러스터 안에 포함된 클라이언트의 수
        self.num_cli = np.random.randint(low=self.num_cli_low, high=self.num_cli_high, size=self.num_clu)
        
        # -- trace --
        # cooked directory에 있는 모든 트레이스 파일
        self.trace_files_all = Env.get_trace_files(TRACE_DIR)
        self.trace_dir = TRACE_DIR
        # trace file 중에서 클러스터 갯수만큼 랜덤으로 trace들을 선택함
        self.selected_traces = self._get_random_traces(self.trace_files_all, self.num_clu)
        # get_remb_of_cluster_head() 함수를 부를 때 마다 bandwidth를 변경하기 위해서 존재함
        self.trace_iterator = 0
        
        # -- video length --
        self.video_length = np.random.randint(low=60, high=300)
        self.counter = 0
        self.end_of_video = False
        
        # -- hardware params --
        self.hd_weight = np.random.randint(10000, 100000)
        
        # -- state --
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.state[0, :self.num_clu] = np.array(self._get_rembs_clu()) # REMB
        self.state[1, :self.num_clu] = np.array(self.num_cli) # the number of clients in each cluster
        self.state[2, -1] = np.array(0) # hardware
        
        return self.state
    
    def step(self, bitrates):
        qoe = 0
        fairness = 0
        hardware = 0
        bandwidth_server = 0
        diff_from_last_bitrate = 0
        
        # QoE
        qoe_quality = 0
        qoe_distortion = 0
        qoe_latency = 0
        
        rembs = self.current_rembs
        for i in range(len(bitrates)):
            remb = rembs[i]
            br = bitrates[i]
            num_of_clients = self.num_cli[i]
            
            qoe_quality += self._quality(br)
            qoe_distortion += self._quality(br) * self._distortion(remb, br)
            qoe_latency += self._quality(br) * (self._latency() / 1000)
             
        qoe = qoe_quality - qoe_distortion - qoe_latency
        
        # fairness
        fairness = self.num_clu * (1 - self._fairness(bitrates, self.current_rembs))
        
        # hardware
        hardware = self._hardware(bitrates)
        
        reward = qoe - fairness - hardware
        
        self.counter += 1
        if self.counter > self.video_length:
            self.end_of_video = True
            self.reset()
            
        # -- state --
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.state[0, :self.num_clu] = np.array(self._get_rembs_clu()) # REMB
        self.state[1, :self.num_clu] = np.array(self.num_cli) # the number of clients in each cluster
        self.state[2, -1] = np.array((sum(bitrates) / self.hd_weight) * 100) # hardware
        
        return self.state, reward, self.end_of_video
        
    def _get_rembs_clu(self):
        """
        부를 때 마다 다음 트레이스를 차례대로 가져옴
        - 일종의 state를 가져오는 함수임.
        """
        idx = self.trace_iterator
        remb_list = []
        for i in range(self.num_clu):
            idx_r = idx % len(self.selected_traces[i])
            bw = self.selected_traces[i][idx_r]
            remb_list.append(bw)
        self.trace_iterator += 1
        self.current_rembs = remb_list
        return remb_list
    
    def _get_random_traces(self, traces, size):
        """
        trace들과 size를 지정하면 trace들에서 랜덤으로 size 갯수만큼 trace를 뽑음
        """
        random_trace_files = np.random.choice(traces, size=size, replace=False)
        self.selected_trace_files = random_trace_files
        
        traces = []
        for path in random_trace_files:
            bws = []
            with open(path, 'rb') as f:
                for line in f:
                    line = line.decode()
                    throughput = int(line)
                    bws.append(throughput * 8 / 1000) # Convert from Byte to Kbps
            traces.append(bws)
                    
        return traces
    
    def _fairness(self, streams, rembs):
        """
        rebms가 0인 경우는 무시하고 계산.
        """
        frac = []
        for i in range(len(streams)):
            if rembs[i] != 0:
                frac.append(streams[i] / rembs[i])

        a = sum(frac) ** 2
        b = len(frac) * sum(list(map(lambda x: x**2, frac)))
        
        if sum(frac) == 0:
            return 1

        return a / b
    
    def _quality(self, q):
        if q <= 0:
            return 0
        else:
            return np.log2(q)
    
    def _distortion(self, remb, bitrate):
        if remb >= bitrate:
            return 0
        else:
            if remb == 0:
                return 0
            else:
                return (bitrate - remb) / remb
        
    def _latency(self):
        return 80
    
    def _hardware(self, streams):
        cpu_usage = (sum(streams) / self.hd_weight) * 100
        hardware = 0 if cpu_usage - 100 < 0 else cpu_usage - 100
        return hardware