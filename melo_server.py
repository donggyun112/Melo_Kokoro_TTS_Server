import json, logging, queue, threading, time, traceback, zmq
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
from melo.api import TTS
from melo.utils import get_text_for_tts_infer
from scipy.signal import resample_poly
from io import BytesIO

# ──────────────────────────────  로깅  ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MeloTTSServer")

# ───────────────────────  오디오 변환 클래스  ───────────────────────
class AudioConverter:
    """오디오 형식과 샘플레이트를 변환하는 클래스"""
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, tgt_sr: int) -> np.ndarray:
        """고품질 오디오 리샘플링"""
        if orig_sr == tgt_sr:
            return audio
        g = np.gcd(orig_sr, tgt_sr)
        return resample_poly(audio, tgt_sr // g, orig_sr // g, padtype="line")
    
    @staticmethod
    def float32_to_int16(audio: np.ndarray) -> np.ndarray:
        """float32 (-1.0~1.0) → int16 (-32768~32767) 변환"""
        # 클리핑 방지 정규화 (최대값이 1.0을 넘지 않도록)
        max_val = np.max(np.abs(audio))
        if max_val > 0.98:  # 여유를 두고 정규화
            audio = audio / max_val * 0.98
        # float32 → int16 변환
        return (audio * 32767).astype(np.int16)
    
    @staticmethod
    def chunk_audio(audio: np.ndarray, chunk_size: int, dtype=np.float32) -> list:
        """오디오를 일정 크기 청크로 분할"""
        chunks = []
        for i in range(0, audio.size, chunk_size):
            chunk = audio[i:i+chunk_size]
            # 마지막 청크가 부족하면 무음으로 패딩
            if chunk.size < chunk_size:
                padded = np.zeros(chunk_size, dtype=audio.dtype)
                padded[:chunk.size] = chunk
                chunk = padded
            chunks.append(chunk.astype(dtype))
        return chunks

# ───────────────────────────  서버 클래스  ───────────────────────────
class MeloTTSServer:
    # 기본값: 24kHz 샘플레이트, 256 샘플 청크 크기
    DEFAULT_SR = 24_000
    DEFAULT_CHUNK_SAMPLES = 1024
    
    def __init__(self, *, max_workers: int = 4, device: str = "mps"):
        self.device = device
        self.models: Dict[str, TTS] = {}
        self.voices: Dict[str, int] = {}
        
        # ── 기본 모델(KR) 로드 ─────────────────────────────────────
        logger.info(f"[SERVER] Loading KR model on {device}")
        self._load_model("KR")
        self.model_sr = self.models["KR"].hps.data.sampling_rate
        logger.info(f"[SERVER] model_sr={self.model_sr}, default_sr={self.DEFAULT_SR}")
        
        # ── ZeroMQ 소켓 ───────────────────────────────────────────
        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.REP)
        self.cmd_sock.bind("tcp://*:5555")
        self.audio_sock = self.ctx.socket(zmq.PUSH)
        self.audio_sock.bind("tcp://*:5556")
        self.poller = zmq.Poller()
        self.poller.register(self.cmd_sock, zmq.POLLIN)
        
        # ── 작업 관리 ─────────────────────────────────────────────
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.stop_flag = threading.Event()
    
    # ────────────────────────────  모델 로더  ─────────────────────────
    def _load_model(self, lang: str) -> bool:
        """지정된 언어의 TTS 모델 로드"""
        if lang in self.models:
            return True
        try:
            tts = TTS(language=lang, device=self.device)
            self.models[lang] = tts
            for name, idx in tts.hps.data.spk2id.items():
                self.voices[name] = idx
            logger.info(f"[SERVER] {lang} voices → {list(self.voices.keys())}")
            return True
        except Exception:
            logger.exception(f"[SERVER] Failed loading model {lang}")
            return False
    
    # ────────────────────────────  모델 워밍업  ─────────────────────────
    def _warmup(self):
        """모델 워밍업 - 첫 번째 추론을 미리 수행하여 성능 향상"""
        logger.info("[SERVER] 모델 워밍업 시작...")
        
        try:
            # 각 로드된 모델마다 워밍업 수행
            for lang, tts in self.models.items():
                # 언어별 적절한 워밍업 텍스트 선택
                if lang == "KR":
                    warmup_text = "안녕하세요. 이것은 모델 워밍업을 위한 텍스트입니다. 한국어 음성 합성 워밍업 테스트입니다."
                elif lang == "EN":
                    warmup_text = "Hello. This is a text for model warmup. This is an English TTS warmup test."
                elif lang == "JP":
                    warmup_text = "こんにちは。これはモデルのウォームアップ用のテキストです。日本語音声合成のウォームアップテストです。"
                elif lang == "ZH":
                    warmup_text = "你好。这是模型预热的文本。中文语音合成预热测试。"
                else:
                    warmup_text = "Hello. This is a text for model warmup."
                
                # 대표 화자 선택 (첫 번째 화자)
                speaker = next(iter(tts.hps.data.spk2id.items()))[1]
                
                # 워밍업 실행
                logger.info(f"[SERVER] {lang} 모델 워밍업: '{warmup_text[:30]}...'")
                
                # 텍스트 처리
                bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                    warmup_text, tts.language, tts.hps, tts.device, tts.symbol_to_id
                )
                speakers = torch.LongTensor([speaker]).to(tts.device)
                
                # 워밍업 추론 수행 (속도 향상을 위해 작은 길이로)
                with torch.no_grad():
                    start_time = time.time()
                    out = tts.model.infer(
                        phones.unsqueeze(0).to(tts.device),
                        torch.LongTensor([phones.size(0)]).to(tts.device),
                        speakers,
                        tones.unsqueeze(0).to(tts.device),
                        lang_ids.unsqueeze(0).to(tts.device),
                        bert.unsqueeze(0).to(tts.device),
                        ja_bert.unsqueeze(0).to(tts.device),
                        sdp_ratio=0.2, noise_scale=0.6,
                        noise_scale_w=0.8, length_scale=1.0
                    )
                    elapsed = time.time() - start_time
                
                # 리소스 해제
                del bert, ja_bert, phones, tones, lang_ids, speakers
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"[SERVER] {lang} 모델 워밍업 완료: {elapsed:.2f}초 소요")
                
            logger.info("[SERVER] 모든 모델 워밍업 완료")
            return True
            
        except Exception as e:
            logger.exception("[SERVER] 워밍업 중 오류 발생")
            return False
    
    # ────────────────────────────  유틸리티  ─────────────────────────
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
    def _worker(self, req: Dict[str, Any], cancel_ev: threading.Event, q: queue.Queue):
        """오디오 생성 작업 처리"""
        job_id = req["job_id"]
        text = req["text"]
        voice = req.get("voice", "KR")
        speed = float(req.get("speed", 1.0))
        # 클라이언트가 요청한 샘플레이트 (없으면 기본값)
        target_sr = int(req.get("target_sample_rate", self.DEFAULT_SR))
        chunk_size = int(req.get("chunk_size", self.DEFAULT_CHUNK_SAMPLES))
        
        # 항상 int16 형식으로 출력하도록 강제 (치지직거림 문제 해결)
        sample_format = "int16"
        
        if voice not in self.voices:
            # 요청된 화자가 없으면 기본 화자 사용
            logger.warning(f"[WORKER] {job_id}: Unknown voice '{voice}', using 'KR'")
            voice = "KR"
        
        # 언어 코드 추출 및 모델 로드
        lang = voice.split("-")[0] if "-" in voice else voice
        if lang not in self.models and not self._load_model(lang):
            logger.error(f"[WORKER] {job_id}: Failed to load {lang} model")
            q.put(("error", f"{lang} model load failed"))
            return
        
        # 모델 및 화자 ID 가져오기
        tts = self.models[lang]
        spk = self.voices[voice]
        
        # 샘플레이트 관련 정보 로깅
        logger.info(f"[WORKER] {job_id}: model_sr={self.model_sr}, "
                   f"target_sr={target_sr}, format={sample_format}")
        
        # 메타 프레임 전송
        meta = {
            "sample_rate": target_sr,
            "format": "pcm",
            "channels": 1,
            "sample_format": sample_format
        }
        self._send_audio(job_id, b"meta", json.dumps(meta).encode())
        
        # 문장 분리 및 처리
        sentences = tts.split_sentences_into_pieces(text, tts.language)
        logger.info(f"[WORKER] {job_id}: Processing {len(sentences)} sentences")
        
        for s_idx, sent in enumerate(sentences, 1):
            if cancel_ev.is_set():
                logger.info(f"[WORKER] {job_id}: Cancelled at sentence #{s_idx}")
                break
            
            logger.debug(f"[WORKER] {job_id}: Processing sentence #{s_idx}: {sent[:30]}...")
            
            try:
                # MeloTTS 텍스트 처리
                bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                    sent, tts.language, tts.hps, tts.device, tts.symbol_to_id
                )
                speakers = torch.LongTensor([spk]).to(tts.device)
                
                # 오디오 생성
                with torch.no_grad():
                    out = tts.model.infer(
                        phones.unsqueeze(0).to(tts.device),
                        torch.LongTensor([phones.size(0)]).to(tts.device),
                        speakers,
                        tones.unsqueeze(0).to(tts.device),
                        lang_ids.unsqueeze(0).to(tts.device),
                        bert.unsqueeze(0).to(tts.device),
                        ja_bert.unsqueeze(0).to(tts.device),
                        sdp_ratio=0.2, noise_scale=0.6,
                        noise_scale_w=0.8, length_scale=1.0/speed
                    )
                
                # 리소스 해제
                del bert, ja_bert, phones, tones, lang_ids, speakers
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # 오디오 추출 및 리샘플링
                audio = out[0][0, 0].cpu().numpy()
                audio = AudioConverter.resample(audio, self.model_sr, target_sr)
                
                # 항상 int16 PCM으로 변환 (치지직거림 문제 해결)
                audio_int16 = AudioConverter.float32_to_int16(audio)
                
                # 오디오 정보 로깅
                logger.debug(f"[WORKER] {job_id}: Audio info: "
                            f"samples={audio_int16.size}, "
                            f"min={audio_int16.min()}, max={audio_int16.max()}, "
                            f"mean={audio_int16.mean():.1f}, dtype={audio_int16.dtype}")
                
                # 청크 분할 및 전송 (int16 포맷으로)
                chunks = AudioConverter.chunk_audio(audio_int16, chunk_size, dtype=np.int16)
                for chunk in chunks:
                    if cancel_ev.is_set():
                        break
                    q.put(chunk.tobytes())
                
                # 문장 처리 완료 로그
                logger.debug(f"[WORKER] {job_id}: Sentence #{s_idx} complete: "
                            f"{len(chunks)} chunks sent")
                
            except Exception as e:
                logger.exception(f"[WORKER] {job_id}: Error processing sentence #{s_idx}")
                q.put(("error", str(e)))
                break
        
        # 작업 완료 신호
        q.put(None)
        logger.info(f"[WORKER] {job_id}: Processing complete")
    
    # ────────────────────────────  송신 스레드  ───────────────────────
    def _sender(self, jid: str, q: queue.Queue):
        """오디오 데이터 전송 스레드"""
        logger.info(f"[SENDER] {jid}: Started sender thread")
        
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
                
                # 오류 신호
                if isinstance(item, tuple) and item[0] == "error":
                    logger.error(f"[SENDER] {jid}: Error - {item[1]}")
                    self._send_audio(jid, b"error", item[1].encode())
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
            
            # 송신 스레드 시작
            threading.Thread(target=self._sender, args=(jid, q), daemon=True).start()
            
            # 오디오 생성 작업 실행
            fut = self.executor.submit(self._worker, msg, cancel, q)
            
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
            voices = list(self.voices.keys())
            self.cmd_sock.send_json({
                "status": "success", 
                "voices": voices
            })
            logger.info(f"[SERVER] Listed {len(voices)} voices")
            
        elif cmd == "load_model":
            # 모델 로드 요청
            lang = msg.get("language")
            if not lang:
                self.cmd_sock.send_json({
                    "status": "error",
                    "message": "Language code required"
                })
                return
                
            ok = self._load_model(lang)
            self.cmd_sock.send_json({
                "status": "success" if ok else "error",
                "message": f"{lang} model {'loaded' if ok else 'load failed'}"
            })
            logger.info(f"[SERVER] Model load: {lang} - {'success' if ok else 'failed'}")
            
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
        logger.info(f"[SERVER] MeloTTS server started (sample_rate={self.DEFAULT_SR}Hz)")
        
        # 워밍업 수행
        self._warmup()
        
        try:
            while not self.stop_flag.is_set():
                # 소켓 폴링
                if dict(self.poller.poll(100)).get(self.cmd_sock):
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
    # 사용 가능한 최적의 디바이스 선택
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available() else "cpu")
    logger.info(f"[SERVER] Using device: {dev}")
    MeloTTSServer(device=dev).start()