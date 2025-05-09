import json, logging, queue, threading, time, traceback, zmq
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, Union, List
from abc import ABC, abstractmethod
from utils import AudioConverter
import numpy as np
from utils.tts import MeloTTSAdapter, KokoroTTSAdapter, TTSModelAdapter
from loguru import logger
import random

class UnifiedTTSServer:
	"""다양한 TTS 모델을 지원하는 통합 TTS 서버"""
	
	def __init__(self, *, model_adapter: TTSModelAdapter, max_workers: int = 4):
		# 모델 어댑터 설정
		self.model_adapter = model_adapter
		
		# 서버 기본 설정
		self.DEFAULT_SR = model_adapter.default_sample_rate
		self.DEFAULT_CHUNK_SAMPLES = 1024  # 1초 분할
		
		# ZMQ 소켓 설정
		self.ctx = zmq.Context()
		self.cmd_sock = self.ctx.socket(zmq.REP)
		self.cmd_sock.bind("tcp://*:5555")
		self.audio_sock = self.ctx.socket(zmq.PUSH)
		self.audio_sock.bind("tcp://*:5556")
		self.poller = zmq.Poller()
		self.poller.register(self.cmd_sock, zmq.POLLIN)
		
		# 스레드풀 설정
		self.executor = ThreadPoolExecutor(max_workers=max_workers)
		
		# 작업 관리
		self.jobs: Dict[str, Dict[str, Any]] = {}
		self.stop_flag = threading.Event()

		self.last_request_time = 0
		self.greeting_timeout = 4.0
		self.lock = threading.Lock()
		
		logger.info(f"[SERVER] Initialized with {model_adapter.model_name}, "
				   f"sample_rate={self.DEFAULT_SR}Hz")
	
	def initialize(self) -> bool:
		"""서버 초기화 (모델 초기화 포함)"""
		# 모델 어댑터 초기화
		if not self.model_adapter.initialize():
			logger.error("[SERVER] Failed to initialize model adapter")
			return False
		
		# 모델 워밍업
		if not self.model_adapter.warmup():
			logger.warning("[SERVER] Model warmup failed, continuing anyway")
		
		logger.info(f"[SERVER] Initialization complete: {self.model_adapter.model_name}")
		return True
	
	def _get_prepared_greeting_kr(self, target_sr: int) -> np.ndarray:
		"""한국어 준비된 인사말 오디오 가져오기"""
		try:
			dir = "./greetings"
			path = f"{dir}/korean_greeting{random.randint(0, 3)}.wav"
			return AudioConverter.load_wav(path, target_sr)
		except Exception as e:
			logger.exception(f"[SERVER] Error loading Korean greeting: {e}")
			return np.zeros(1, dtype=np.float32)  # 실패 시 빈 오디오

	def _get_prepared_greeting_en(self, target_sr: int) -> np.ndarray:
		"""영어 준비된 인사말 오디오 가져오기"""
		try:
			dir = "./greetings"
			path = f"{dir}/english_greeting{random.randint(0, 3)}.wav"
			return AudioConverter.load_wav(path, target_sr)
		except Exception as e:
			logger.exception(f"[SERVER] Error loading English greeting: {e}")
			return np.zeros(1, dtype=np.float32)  # 실패 시 빈 오디오

	def _send_prepared_audio(self, audio: np.ndarray, target_sr: int, chunk_size: int, q: queue.Queue):
		"""준비된 오디오를 큐에 전송"""
		try:
			# float32 → int16 변환
			audio_int16 = AudioConverter.float32_to_int16(audio)
			
			# 청크로 분할 및 큐에 추가
			chunks = AudioConverter.chunk_audio(audio_int16, chunk_size, dtype=np.int16)
			for chunk in chunks:
				q.put(chunk.tobytes())
				
			logger.debug(f"[SERVER] Sent prepared greeting: {len(chunks)} chunks")
		except Exception as e:
			logger.exception(f"[SERVER] Error sending prepared greeting: {e}")
	def _send_audio(self, jid: str, mtype: bytes, data: bytes):
		"""오디오 데이터 전송"""
		try:
			self.audio_sock.send_multipart([jid.encode(), mtype, data])
			# 디버깅용 로그 (data 크기에 따라 로그 레벨 조정)
			if mtype == b"meta":
				logger.debug(f"[SENDER] {jid}: meta → {data.decode()}")
			elif mtype == b"data":
				logger.debug(f"[SENDER] {jid}: data → {len(data)} bytes")
			elif mtype == b"end":
				logger.info(f"[SENDER] {jid}: end → {data.decode()}")
			elif mtype == b"error":
				logger.error(f"[SENDER] {jid}: error → {data.decode()}")
		except Exception:
			logger.exception(f"[SENDER] socket send failed for {jid}")
	
	# ────────────────────────────  워커  ────────────────────────────
	def _worker(self, req: Dict[str, Any], cancel_ev: threading.Event, q: queue.Queue, play_greeting: bool = True):
		"""오디오 생성 작업 처리"""
		job_id = req["job_id"]
		text = req["text"]
		voice = req.get("voice", "KR")  # 기본 화자는 모델에 따라 다름
		speed = float(req.get("speed", 1.0))
		target_sr = int(req.get("target_sample_rate", self.DEFAULT_SR))
		chunk_size = int(req.get("chunk_size", self.DEFAULT_CHUNK_SAMPLES))
		
		# 항상 int16 형식으로 출력하도록 강제
		sample_format = "int16"
		
		# 메타 프레임 전송
		meta = {
			"sample_rate": target_sr,
			"format": "pcm",
			"channels": 1,
			"sample_format": sample_format
		}
		self._send_audio(job_id, b"meta", json.dumps(meta).encode())
		
		try:
			# 타임아웃에 따라 인사말 출력 여부 결정
			if play_greeting:
				if isinstance(self.model_adapter, MeloTTSAdapter):
					# MeloTTS는 한국어 인사말 먼저 보내기
					prepared_audio = self._get_prepared_greeting_kr(target_sr)
					self._send_prepared_audio(prepared_audio, target_sr, chunk_size, q)
					logger.info(f"[WORKER] {job_id}: Sent Korean greeting")
				elif isinstance(self.model_adapter, KokoroTTSAdapter):
					# KokoroTTS는 영어 인사말 먼저 보내기
					prepared_audio = self._get_prepared_greeting_en(target_sr)
					self._send_prepared_audio(prepared_audio, target_sr, chunk_size, q)
					logger.info(f"[WORKER] {job_id}: Sent English greeting")
			else:
				logger.info(f"[WORKER] {job_id}: Skipped greeting (within timeout)")
			
			# 중단 요청 확인
			if cancel_ev.is_set():
				logger.info(f"[WORKER] {job_id}: Cancelled before audio generation")
				q.put(("end", "interrupted"))
				return
			
			# 모델 어댑터를 통해 오디오 생성
			# req에서 중복 파라미터 필터링 (text 등이 중복으로 전달되는 문제 해결)
			filtered_req = {k: v for k, v in req.items() 
						   if k not in ['text', 'voice', 'speed', 'target_sample_rate']}
			
			audio, actual_sr = self.model_adapter.generate_audio(
				text=text,
				voice=voice,
				speed=speed,
				target_sr=target_sr,
				**filtered_req  # 필터링된 추가 파라미터만 전달
			)
			
			# 중단 요청 확인
			if cancel_ev.is_set():
				q.put(("end", "interrupted"))
				return
			
			# 샘플레이트 불일치 확인
			if actual_sr != target_sr:
				logger.warning(f"[WORKER] {job_id}: Sample rate mismatch: {actual_sr} vs {target_sr}")
				audio = AudioConverter.resample(audio, actual_sr, target_sr)
			
			# float32 → int16 변환 (필요한 경우)
			if audio.dtype == np.float32:
				audio_int16 = AudioConverter.float32_to_int16(audio)
			elif audio.dtype == np.int16:
				audio_int16 = audio
			else:
				# 다른 형식이면 float32로 변환 후 int16으로 변환
				logger.warning(f"[WORKER] {job_id}: Unexpected audio format: {audio.dtype}")
				audio_float32 = audio.astype(np.float32)
				if audio_float32.max() > 1.0 or audio_float32.min() < -1.0:
					audio_float32 = audio_float32 / np.max(np.abs(audio_float32)) * 0.98
				audio_int16 = AudioConverter.float32_to_int16(audio_float32)
			
			# 오디오 정보 로깅
			logger.debug(f"[WORKER] {job_id}: Audio info: "
						f"samples={audio_int16.size}, "
						f"min={audio_int16.min()}, max={audio_int16.max()}, "
						f"mean={audio_int16.mean():.1f}, dtype={audio_int16.dtype}")
			
			# 청크 분할 및 전송 (int16 포맷으로)
			chunks = AudioConverter.chunk_audio(audio_int16, chunk_size, dtype=np.int16)
			for chunk in chunks:
				if cancel_ev.is_set():
					logger.info(f"[WORKER] {job_id}: Cancelled during chunk transmission")
					q.put(("end", "interrupted"))
					return
				q.put(chunk.tobytes())
			
			# 작업 완료 신호
			q.put(None)
			
		except Exception as e:
			logger.exception(f"[WORKER] {job_id}: Error processing text")
			q.put(("error", str(e)))
	
	# ────────────────────────────  송신 스레드  ───────────────────────
	def _sender(self, jid: str, q: queue.Queue):
		"""오디오 데이터 전송 스레드"""
		try:
			total_chunks = 0
			total_bytes = 0
			start_time = time.time()
			
			while True:
				item = q.get()
				
				# 완료 신호
				if item is None:
					elapsed = time.time() - start_time
					logger.info(f"[SENDER] {jid}: Complete - sent {total_chunks} chunks, "
								f"{total_bytes/1024:.1f} KB in {elapsed:.2f}s")
					self._send_audio(jid, b"end", b"completed")
					break
				
				# 오류 또는 중단 신호
				if isinstance(item, tuple):
					if item[0] == "error":
						logger.error(f"[SENDER] {jid}: Error - {item[1]}")
						self._send_audio(jid, b"error", item[1].encode())
					elif item[0] == "end":
						logger.info(f"[SENDER] {jid}: {item[1]}")
						self._send_audio(jid, b"end", item[1].encode())
					break
				
				# 오디오 데이터 전송
				self._send_audio(jid, b"data", item)
				total_chunks += 1
				total_bytes += len(item)
				
		except Exception:
			logger.exception(f"[SENDER] {jid}: Unexpected error")
			self._send_audio(jid, b"error", b"Sender thread error")
		finally:
			# 작업 완료 시 목록에서 제거
			self.jobs.pop(jid, None)
			logger.info(f"[SENDER] {jid}: Sender thread terminated")
	
	# ────────────────────────────  명령 처리  ────────────────────────
	def _process_cmd(self, msg: Dict[str, Any]):
		"""클라이언트 명령 처리"""
		cmd = msg.get("command")
		jid = msg.get("job_id")
		
		if cmd == "generate":
			# 음성 생성 요청
			self.cmd_sock.send_json({"status": "started", "job_id": jid})
			q = queue.Queue()
			cancel = threading.Event()
			
			# 현재 시간과 마지막 요청 시간 비교하여 인사말 출력 여부 결정
			current_time = time.time()
			should_play_greeting = False
			
			with self.lock:
				if current_time - self.last_request_time > self.greeting_timeout:
					should_play_greeting = True
				self.last_request_time = current_time
			
			# 송신 스레드 시작
			threading.Thread(target=self._sender, args=(jid, q), daemon=True).start()
			
			# 오디오 생성 작업 실행 (인사말 플래그 전달)
			fut = self.executor.submit(self._worker, msg, cancel, q, should_play_greeting)
			
			# 작업 추적
			self.jobs[jid] = {"cancel": cancel, "fut": fut}
			logger.info(f"[SERVER] Started job {jid}")
			
		elif cmd == "interrupt":
			# 작업 중단 요청
			interrupted = False
			
			if jid and jid in self.jobs:
				# 특정 작업 중단
				self.jobs[jid]["cancel"].set()
				interrupted = True
				logger.info(f"[SERVER] Interrupted job {jid}")
			elif jid is None:
				# 모든 작업 중단
				for j in self.jobs.values():
					j["cancel"].set()
				interrupted = len(self.jobs) > 0
				logger.info(f"[SERVER] Interrupted all jobs ({len(self.jobs)})")
			
			self.cmd_sock.send_json({
				"status": "interrupted" if interrupted else "not_found",
				"job_id": jid
			})
			
		elif cmd == "list_voices":
			# 음성 목록 요청
			voices = self.model_adapter.list_voices()
			self.cmd_sock.send_json({
				"status": "success", 
				"voices": voices
			})
			logger.info(f"[SERVER] Listed {len(voices)} voices")
			
		else:
			# 알 수 없는 명령
			self.cmd_sock.send_json({
				"status": "error", 
				"message": "unknown command"
			})
			logger.warning(f"[SERVER] Unknown command: {cmd}")
	
	# ─────────────────────────────  메인 루프  ────────────────────────
	def start(self):
		"""서버 시작 및 명령 처리"""
		if not self.initialize():
			logger.error("[SERVER] Failed to initialize server. Exiting.")
			return
			
		logger.info(f"[SERVER] {self.model_adapter.model_name} server started (sample_rate={self.DEFAULT_SR}Hz)")
		
		try:
			while not self.stop_flag.is_set():
				# 소켓 폴링
				if dict(self.poller.poll(1000)).get(self.cmd_sock):
					try:
						msg = self.cmd_sock.recv_json()
						self._process_cmd(msg)
					except Exception:
						logger.exception("[SERVER] Command processing error")
						try:
							self.cmd_sock.send_json({
								"status": "error",
								"message": "Internal server error"
							})
						except:
							pass
		except KeyboardInterrupt:
			logger.info("[SERVER] Received keyboard interrupt")
		except Exception:
			logger.exception("[SERVER] Unexpected error in main loop")
		finally:
			self._cleanup()
	
	# ─────────────────────────────  종료 처리  ────────────────────────
	def _cleanup(self):
		"""서버 종료 및 리소스 정리"""
		logger.info("[SERVER] Cleaning up resources")
		
		# 모든 작업 취소
		for j in self.jobs.values():
			j["cancel"].set()
		
		# 스레드풀 종료
		self.executor.shutdown(cancel_futures=True)
		
		# ZMQ 소켓 정리
		self.poller.unregister(self.cmd_sock)
		self.cmd_sock.close()
		self.audio_sock.close()
		self.ctx.term()
		
		logger.info("[SERVER] Server shutdown complete")

# ─────────────────────────────  실행부  ─────────────────────────────
if __name__ == "__main__":
	import argparse
	
	# 명령행 인수 파싱
	parser = argparse.ArgumentParser(description="Unified TTS Server")
	parser.add_argument("--model", type=str, default="melo", choices=["melo", "kokoro"],
					  help="TTS model to use (default: melo)")
	parser.add_argument("--device", type=str, default="cpu",
					  help="Device for MeloTTS (cpu, cuda, mps)")
	parser.add_argument("--kokoro-model", type=str, default="./KoKoro_models/kokoro-v1.0.onnx",
					  help="Path to Kokoro model file")
	parser.add_argument("--kokoro-voices", type=str, default="./KoKoro_models/voices-v1.0.bin",
					  help="Path to Kokoro voices file")
	parser.add_argument("--workers", type=int, default=4,
					  help="Number of worker threads")
	parser.add_argument("--debug", action="store_true",
					  help="Enable debug logging")
	
	args = parser.parse_args()
	
	# 디버그 모드 설정
	if args.debug:
		logger.setLevel(logging.DEBUG)
	
	# 모델 어댑터 생성
	if args.model == "melo":
		# 사용 가능한 최적의 디바이스 선택 (명시적 지정이 없을 경우)
		if args.device == "cpu":
			import torch
			dev = ("cuda" if torch.cuda.is_available() else
				  "mps" if getattr(torch.backends, "mps", None)
						   and torch.backends.mps.is_available() else "cpu")
			logger.info(f"[SERVER] Auto-detected device: {dev}")
		else:
			dev = args.device
			
		model_adapter = MeloTTSAdapter(device=dev)
	else:  # kokoro
		model_adapter = KokoroTTSAdapter(
			model_path=args.kokoro_model,
			voice_path=args.kokoro_voices
		)
	
	# 서버 생성 및 시작
	server = UnifiedTTSServer(model_adapter=model_adapter, max_workers=args.workers)
	server.start()