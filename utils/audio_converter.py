import numpy as np
from loguru import logger
from scipy.io import wavfile

class AudioConverter:    
	@staticmethod
	def resample(audio: np.ndarray, orig_sr: int, tgt_sr: int) -> np.ndarray:
		if orig_sr == tgt_sr:
			return audio
			
		try:
			from scipy.signal import resample_poly
			g = np.gcd(orig_sr, tgt_sr)
			return resample_poly(audio, tgt_sr // g, orig_sr // g, padtype="line")
		except ImportError:
			new_len = int(len(audio) * tgt_sr / orig_sr)
			return np.interp(
				np.linspace(0, len(audio) - 1, new_len),
				np.arange(len(audio)),
				audio
			)
	
	@staticmethod
	def float32_to_int16(audio: np.ndarray) -> np.ndarray:
		max_val = np.max(np.abs(audio))
		if max_val > 0.98:
			audio = audio / max_val * 0.98
		return (audio * 32767).astype(np.int16)
	
	@staticmethod
	def int16_to_float32(audio: np.ndarray) -> np.ndarray:
		return audio.astype(np.float32) / 32767.0
	
	@staticmethod
	def chunk_audio(audio: np.ndarray, chunk_size: int, dtype=np.float32) -> list:
		chunks = []
		for i in range(0, audio.size, chunk_size):
			chunk = audio[i:i+chunk_size]
			if chunk.size < chunk_size:
				padded = np.zeros(chunk_size, dtype=audio.dtype)
				padded[:chunk.size] = chunk
				chunk = padded
			chunks.append(chunk.astype(dtype))
		return chunks
	
	@staticmethod
	def load_wav(filepath: str, target_sr: int) -> np.ndarray:
		"""WAV 파일 로드 및 리샘플링"""
		sr, data = wavfile.read(filepath)
		if sr != target_sr:
			data = AudioConverter.resample(data, sr, target_sr)
		return data