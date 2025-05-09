import zmq
import threading
import numpy as np
import sounddevice as sd
import json
import time
import uuid

class MeloTTSClient:
    def __init__(self, server_address="localhost"):
        # ZeroMQ 설정
        self.context = zmq.Context()
        
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{server_address}:5555")
        
        self.audio_socket = self.context.socket(zmq.PULL)
        self.audio_socket.connect(f"tcp://{server_address}:5556")
        
        # MeloTTS의 기본 샘플레이트로 변경
        self.sample_rate = 22050  # MeloTTS 기본값
        self.stop_flag = False
        
        self.current_job_id = None
        
        self.audio_stream = None
        
        self.receiver_thread = threading.Thread(target=self._receive_audio)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
    
    def _receive_audio(self):
        """오디오 데이터 수신 스레드"""
        print("오디오 수신 스레드 시작")
        
        while not self.stop_flag:
            try:
                try:
                    parts = self.audio_socket.recv_multipart(flags=zmq.NOBLOCK)
                    print(f"메시지 수신: {len(parts)} 부분")
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                
                if len(parts) != 3:
                    print(f"잘못된 메시지 형식: {len(parts)} 부분")
                    continue
                
                job_id, msg_type, data = parts
                job_id = job_id.decode('utf-8')
                
                if job_id != self.current_job_id:
                    print(f"다른 작업({job_id})의 메시지 무시 (현재 작업: {self.current_job_id})")
                    continue
                
                if msg_type == b'meta':
                    meta_data = json.loads(data.decode('utf-8'))
                    self.sample_rate = meta_data.get('sample_rate', 24000)
                    print(f"메타데이터 수신: 샘플레이트={self.sample_rate}")
                    
                    if self.audio_stream is not None:
                        self.audio_stream.stop()
                        self.audio_stream.close()
                    
                    self.audio_stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype='float32'
                    )
                    self.audio_stream.start()
                    
                elif msg_type == b'data':
                    if self.audio_stream is None:
                        print("오디오 데이터 수신했으나 스트림이 준비되지 않음")
                        continue
                    
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    print(f"오디오 데이터 수신: {len(audio_data)} 샘플")
                    
                    try:
                        self.audio_stream.write(audio_data)
                        print(f"오디오 데이터 재생: {len(audio_data)} 샘플")
                    except Exception as e:
                        print(f"오디오 재생 오류: {e}")
                
                elif msg_type == b'end':
                    status = data.decode('utf-8')
                    print(f"작업 종료: {status}")
                    if status == 'interrupted':
                        print("작업이 인터럽트되었습니다.")
                
                elif msg_type == b'error':
                    error_msg = data.decode('utf-8')
                    print(f"오류 발생: {error_msg}")
            
            except Exception as e:
                print(f"오디오 수신 중 오류: {e}")
                time.sleep(0.1)
    
    def generate_speech(self, text, voice="KR", speed=1.2):
        """텍스트에서 음성 생성 요청"""
        # 새 작업 ID 생성
        job_id = str(uuid.uuid4())
        self.current_job_id = job_id
        
        # 생성 요청 전송
        message = {
            'command': 'generate',
            'job_id': job_id,
            'text': text,
            'voice': voice,
            'speed': speed
        }
        
        print(f"음성 생성 요청: {text[:30]}...")
        self.command_socket.send_json(message)
        response = self.command_socket.recv_json()
        
        if response.get('status') != 'started':
            print(f"요청 오류: {response}")
            return None
        
        print(f"음성 생성 시작됨: 작업 ID {job_id}")
        return job_id
    
    def interrupt(self, job_id=None):
        """현재 또는 지정된 작업 인터럽트"""
        if job_id is None:
            job_id = self.current_job_id
        
        if job_id is None:
            print("인터럽트할 작업이 없습니다.")
            return False
        
        # 인터럽트 요청 전송
        message = {
            'command': 'interrupt',
            'job_id': job_id
        }
        
        print(f"작업 인터럽트 요청: {job_id}")
        self.command_socket.send_json(message)
        response = self.command_socket.recv_json()
        
        if response.get('status') == 'interrupted':
            print(f"작업 {job_id} 인터럽트 성공")
            return True
        else:
            print(f"작업 {job_id} 인터럽트 실패: {response}")
            return False
    
    def list_voices(self):
        """사용 가능한 음성 목록 조회"""
        message = {'command': 'list_voices'}
        self.command_socket.send_json(message)
        response = self.command_socket.recv_json()
        
        if response.get('status') == 'success':
            return response.get('voices', [])
        else:
            print(f"음성 목록 조회 오류: {response}")
            return []
    
    def close(self):
        """클라이언트 종료"""
        self.stop_flag = True
        time.sleep(0.5)
        
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        self.command_socket.close()
        self.audio_socket.close()
        self.context.term()
        print("클라이언트가 종료되었습니다.")

if __name__ == "__main__":
    client = MeloTTSClient()
    
    try:
        voices = client.list_voices()
        print(f"사용 가능한 음성: {voices}")
        
        text1 = "안녕하세요. 반갑습니다. 이 코드는 한국어 MeloTTS 예제입니다. 인터럽트 기능을 테스트합니다.  아주 긴 문장입니다. 이 문장은 음성 생성 후 인터럽트가 발생할 때까지 계속 재생됩니다. 이 문장은 음성 생성 후 인터럽트가 발생할 때까지 계속 재생됩니다."
        job_id1 = client.generate_speech(text1, voice="KR", speed=1.2)
        
        time.sleep(5)
        client.interrupt()
        
        text2 = "두 번째 문장입니다. 인터럽트 후 즉시 재생되는지 확인해보세요."
        job_id2 = client.generate_speech(text2, voice="KR", speed=1.0)
        
        # 다른 언어 테스트 (영어)
        time.sleep(8)
        text3 = "This is an English test. MeloTTS supports multiple languages."
        job_id3 = client.generate_speech(text3, voice="EN", speed=1.0)
        
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")
    finally:
        client.close()