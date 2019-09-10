import os
import numpy as np

RANDOM_SEED = None
RANDOM_SEED = 10

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
        # 각 클러스터 안에 포함된 클라이언트의 수
        self.num_of_client_in_each_cluster = np.random.randint(low=num_of_client_low, high=num_of_client_high, size=self.num_of_cluster)
        
        # -- trace --
        # cooked directory에 있는 모든 트레이스 파일
        self.trace_files_all = Environment.get_trace_files(trace_dir)
        # trace file 중에서 클러스터 갯수만큼 랜덤으로 trace들을 선택함
        self.selected_traces = self.get_random_traces(self.trace_files_all, self.num_of_cluster)
        # get_remb_of_cluster_head() 함수를 부를 때 마다 bandwidth를 변경하기 위해서 존재함
        self.trace_iterator = 0
        
    def get_random_traces(self, traces, size):
        """
        trace들과 size를 지정하면 trace들에서 랜덤으로 size 갯수만큼 trace를 뽑음
        """
        
        random_trace_files = np.random.choice(traces, size=size, replace=False)
        
        traces = []
        for path in random_trace_files:
            bws = []
            with open(path, 'rb') as f:
                for line in f:
                    line = line.decode()
                    throughput = int(line)
                    bws.append(throughput * 8 / 1000000) # Convert from Byte to Mbps
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
        return remb_list
    
    def set_bitrate_of_streams(self, bitrates_of_streams, rembs):
        """
        각 stream을 설정해주면 Utility를 계산해서 되돌려줌.
        """
        qoe = 0
        fairness = 0
        hardware = 0
        bandwidth_server = 0
        diff_from_last_bitrate = 0
        
        # QoE
        for i in range(len(bitrates_of_streams)):
             qoe += self.num_of_client_in_each_cluster[i]
                
        # fairness
        fairness = self.fairness(bitrates_of_streams, rembs)
        
        # hardware
        
        # bandwidth_server
        
        utility = qoe - fairness
        
        return utility
    
    def fairness(self, streams, remb):
        frac = []
        for i in range(len(streams)):
            frac.append(streams[i] / rembs[i])

        a = sum(frac) ** 2
        b = len(frac) * sum(list(map(lambda x: x**2, frac)))

        return a / b