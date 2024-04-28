import numpy as np
import soundcard as sc
import time
import mss
import sentencepiece as spm
import torch
from PIL import Image
import cv2
import sys

SAMPLE_RATE = 19200
REAL_FRAME_RATE = 15
MULTIPLY_FRAMES = 2
FRAME_RATE = REAL_FRAME_RATE * MULTIPLY_FRAMES
FRAMES_PER_CHUNK = 8 # should be divisible on multiply frames
BUFFER_SIZE = 4
CONTEXT_LENGTH = 4
RATE_RATIO = SAMPLE_RATE//FRAME_RATE
NUM_AUDIO_FRAMES = SAMPLE_RATE*FRAMES_PER_CHUNK//FRAME_RATE
MIN_FRAME_TIME=1/(REAL_FRAME_RATE+2)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 340




def mic_stream(q_audio, event, event_keyboar_interrupt):
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
        event.wait()
        while not event_keyboar_interrupt.is_set():
            recording = mic.record(numframes=NUM_AUDIO_FRAMES)
            data = np.mean(recording, axis=1)
            if q_audio.full():
                q_audio.get()
            q_audio.put(torch.tensor(data).unsqueeze(-1))

    return

def screen_stream(q_vid, event, event_keyboar_interrupt):
    sct = mss.mss()
    monitor = sct.monitors[1]
    event.wait()
    while not event_keyboar_interrupt.is_set():
        frames = []
        for _ in range(FRAMES_PER_CHUNK // MULTIPLY_FRAMES):
            start = time.time()
            img = sct.grab(monitor)
            frame = torch.Tensor(cv2.resize(np.array(Image.frombytes('RGB', img.size, img.rgb)), (SCREEN_WIDTH, SCREEN_HEIGHT)))
            frames.extend([frame] * MULTIPLY_FRAMES)
            time.sleep(max(0, MIN_FRAME_TIME - (time.time() - start)))
        if q_vid.full():
            q_vid.get()
        q_vid.put(torch.stack(frames))
    return





import torchaudio
import torchvision
import logging



class ContextCacher:
    def __init__(self, segment_length: int, context_length: int, rate_ratio: int):
        self.segment_length = segment_length
        self.context_length = context_length

        self.context_length_v = context_length
        self.context_length_a = context_length * rate_ratio
        self.context_v = torch.zeros([self.context_length_v, 3, SCREEN_HEIGHT, SCREEN_WIDTH])
        self.context_a = torch.zeros([self.context_length_a, 1])

    def __call__(self, chunk_v, chunk_a):
        if chunk_v.size(0) < self.segment_length:
            chunk_v = torch.nn.functional.pad(chunk_v, (0, 0, 0, 0, 0, 0, 0, self.segment_length - chunk_v.size(0)))
        if chunk_a.size(0) < self.segment_length * 640:
            chunk_a = torch.nn.functional.pad(chunk_a, (0, 0, 0, self.segment_length * 640 - chunk_a.size(0)))
        if self.context_length == 0:
            return chunk_v.float(), chunk_a.float()
        else:
            chunk_with_context_v = torch.cat((self.context_v, chunk_v))
            chunk_with_context_a = torch.cat((self.context_a, chunk_a))
            self.context_v = chunk_v[-self.context_length_v :]
            self.context_a = chunk_a[-self.context_length_a :]
            return chunk_with_context_v.float(), chunk_with_context_a.float()



from avsr.data_prep.detectors.mediapipe.detector import LandmarksDetector
from avsr.data_prep.detectors.mediapipe.video_process import VideoProcess


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class Preprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess()
        self.video_transform = torch.nn.Sequential(
            FunctionalModule(
                lambda n: [(lambda x: torchvision.transforms.functional.resize(x, 44, antialias=True))(i) for i in n]
            ),
            FunctionalModule(lambda x: torch.stack(x)),
            torchvision.transforms.Normalize(0.0, 255.0),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def forward(self, audio, video):
        video = video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video).permute(0, 3, 1, 2).float()
        video = self.video_transform(video)
        audio = audio.mean(axis=-1, keepdim=True)
        return audio, video



class SentencePieceTokenProcessor:
    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens, lstrip: bool = True) -> str:
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        output_string = "".join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace("\u2581", " ")

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string


class InferencePipeline(torch.nn.Module):
    def __init__(self, preprocessor, model, decoder, token_processor):
        super().__init__()
        self.preprocessor = preprocessor
        self.model = model
        self.decoder = decoder
        self.token_processor = token_processor

        self.state = None
        self.hypotheses = None

    def forward(self, audio, video):
        audio, video = self.preprocessor(audio, video)
        feats = self.model(audio.unsqueeze(0), video.unsqueeze(0))
        length = torch.tensor([feats.size(1)], device=audio.device)
        self.hypotheses, self.state = self.decoder.infer(feats, length, 10, state=self.state, hypothesis=self.hypotheses)
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript


def _get_inference_pipeline(model_path, spm_model_path):
    model = torch.jit.load(model_path)
    model.eval()

    sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)
    token_processor = SentencePieceTokenProcessor(sp_model)

    decoder = torchaudio.models.RNNTBeamSearch(model.model, sp_model.get_piece_size())

    return InferencePipeline(
        preprocessor=Preprocessing(),
        model=model,
        decoder=decoder,
        token_processor=token_processor,
    )



from torchaudio.utils import download_asset


def main():
    import torch.multiprocessing as mp
    ctx_audio = mp.get_context("spawn")
    ctx_video = mp.get_context("spawn")
    event = mp.Event()
    event_keyboar_interrupt = mp.Event()
    q_audio = ctx_audio.Queue()
    p_audio = ctx_audio.Process(target=mic_stream, args=(q_audio, event, event_keyboar_interrupt))
    q_video = ctx_video.Queue()
    p_video = ctx_video.Process(target=screen_stream, args=(q_video, event, event_keyboar_interrupt))
    cacher = ContextCacher(BUFFER_SIZE, CONTEXT_LENGTH, RATE_RATIO)

    try:
        print("Building pipeline...")
        model_path = download_asset("tutorial-assets/device_avsr_model.pt")
        spm_model_path = download_asset("tutorial-assets/spm_unigram_1023.model")

        pipeline = _get_inference_pipeline(model_path, spm_model_path)


        @torch.inference_mode()
        def infer():
            num_video_frames = 0
            video_chunks = []
            audio_chunks = []
            while True:
                chunk_v, chunk_a = q_video.get(), q_audio.get()
                num_video_frames += chunk_a.size(0) // NUM_AUDIO_FRAMES
                video_chunks.append(chunk_v)
                audio_chunks.append(chunk_a)
                if num_video_frames < BUFFER_SIZE:
                    continue
                video = torch.cat(video_chunks)
                video = video.permute(0,3,1,2)
                audio = torch.cat(audio_chunks)
                video, audio = cacher(video, audio)
                pipeline.state, pipeline.hypotheses = None, None
                try:
                    transcript = pipeline(audio, video.float())
                    print(transcript, end="", flush=True)
                except AssertionError:
                    print("NO PERSON DETECTED")
                except RuntimeError as re:
                    print("Model Lacks Information", re)
                num_video_frames = 0
                video_chunks = []
                audio_chunks = []
                       
        p_audio.start()
        p_video.start()
        print("Initializing..")
        time.sleep(5)
        event.set()
        infer()
        p_audio.join()
        p_video.join()
    except KeyboardInterrupt:
        event_keyboar_interrupt.set()
        while not q_audio.empty():
            q_audio.get()
        q_audio.get()
        p_audio.join()
        while not q_video.empty():
            q_video.get()
        p_video.join()
        sys.stderr.write("Program terminated by user\n")
        exit(2)
    


if __name__ == "__main__":
    main(
    )

