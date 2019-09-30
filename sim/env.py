import os, sys
import numpy as np
import math

RANDOM_SEED = None

# weighed parameters of QoE metric
RHO = 1

class Environment:
    """
    WebRTC를 이용한 [One server -- Many] 스트리밍 아키텍쳐를 구현함.
    """
    
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
    
    # -- instance method --
    def __init__(self,
                 random_seed=RANDOM_SEED,
                 num_of_cluster_low=1,
                 num_of_cluster_high=100,
                 num_of_client_low=1,
                 num_of_client_high=30,
                 trace_year_month=None,
                 trace_dir="../dataset/cooked/",
                 trace_type="youtube.com"
                ):
        
        # -- random seed --
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        # -- network configuration --
        # 클러스터 갯수를 랜덤으로 지정
        self.num_of_cluster = np.random.randint(low=num_of_cluster_low, high=num_of_cluster_high)
        self.num_of_cluster_low = num_of_cluster_low
        self.num_of_cluster_high = num_of_cluster_high
        # 각 클러스터 안에 포함된 클라이언트의 수
        self.num_of_client_in_each_cluster = np.random.randint(low=num_of_client_low, high=num_of_client_high, size=self.num_of_cluster)
        self.num_of_client_low = num_of_client_low
        self.num_of_client_high = num_of_client_high
        
        # -- trace --
        # cooked directory에 있는 모든 트레이스 파일
        self.trace_files_all = Environment.get_trace_files(trace_dir)
        self.trace_dir = trace_dir
        # trace file 중에서 클러스터 갯수만큼 랜덤으로 trace들을 선택함
        self.selected_traces = self.get_random_traces(self.trace_files_all, self.num_of_cluster)
        # get_remb_of_cluster_head() 함수를 부를 때 마다 bandwidth를 변경하기 위해서 존재함
        self.trace_iterator = 0
        
        self.video_length = np.random.randint(low=60, high=300)
        self.counter = 0
        
    def get_random_traces(self, traces, size):
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
    
    def get_remb_of_cluster_head(self):
        """
        부를 때 마다 다음 트레이스를 차례대로 가져옴
        """
        
        idx = self.trace_iterator
        remb_list = []
        for i in range(self.num_of_cluster):
            idx_r = idx % len(self.selected_traces[i])
            bw = self.selected_traces[i][idx_r]
            remb_list.append(bw)
        self.trace_iterator += 1
        self.current_rembs = remb_list
        return remb_list
    
    def set_bitrate_of_streams(self, bitrates_of_streams):
        """
        각 stream을 설정해주면 Utility를 계산해서 되돌려줌.
        """
        qoe = 0
        fairness = 0
        hardware = 0
        bandwidth_server = 0
        diff_from_last_bitrate = 0
        
        # QoE
        qoe_quality = 0
        qoe_distortion = 0
        qoe_latency = 0
        
        for i in range(len(bitrates_of_streams)):
            remb = self.current_rembs[i]
            br = bitrates_of_streams[i]
            num_of_clients = self.num_of_client_in_each_cluster[i]
            
            qoe_quality += self.quality(br)
            qoe_distortion += self.quality(br) * self.distortion(remb, br)
            qoe_latency += self.quality(br) * (self.latency() / 1000)
             
        qoe = qoe_quality - qoe_distortion - qoe_latency
        
        # fairness
        fairness = self.num_of_cluster * (1 - self.fairness(bitrates_of_streams, self.current_rembs))
        
        # hardware
        
        # bandwidth_server
        
        utility = qoe - fairness
        
        self.end_of_video = False
        self.counter += 1
        if self.counter > self.video_length:
            self.end_of_video = True
            self.reset_all_params()
        
        return utility, self.end_of_video
    
    def reset_all_params(self):
        self.num_of_cluster = np.random.randint(low=self.num_of_cluster_low, high=self.num_of_cluster_high)
        self.num_of_client_in_each_cluster = np.random.randint(low=self.num_of_client_low, high=self.num_of_client_high, size=self.num_of_cluster)
        self.trace_files_all = Environment.get_trace_files(self.trace_dir)
        self.selected_traces = self.get_random_traces(self.trace_files_all, self.num_of_cluster)
        self.trace_iterator = 0
        self.counter = 0
    
    ###
    
    def fairness(self, streams, rembs):
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
    
    def quality(self, q):
        if q <= 0:
            return 0
        else:
            return np.log2(q)
    
    def distortion(self, remb, bitrate):
        if remb >= bitrate:
            return 0
        else:
            if remb == 0:
                return 0
            else:
                return (bitrate - remb) / remb
        
    def latency(self):
        return 80
    
    def hardware(self, streams):
        return sum(streams)**1.5
    
    