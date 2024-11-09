import os

from src import log
from src.utils import utils

import asyncio
import whisper
from whisper import DecodingResult

from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as transformers_log
from huggingface_hub import hf_hub_download, login

import torch
import torchaudio
import torch.quantization

class InferenceService:
    _initialized = False

    def __init__(self, language='it', num_threads=1, quantization=True, device = "cpu"):
        try:
            login(token=os.environ['API_TOKEN'])
            log.info("HuggingFace login successful")
        except Exception as e:
            log.error(f"Error during HuggingFace login: {str(e)}")
            raise

        if not InferenceService._initialized:
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            transformers_log.set_verbosity_error()
            self.model_name = utils.MERGED_MODEL_NAME
            self.language = language
            self.pytorch_converted_model_source = utils.PRETRAINED_MODEL_PTH
            self.pytorch_converted_model_filename = utils.PRETRAINED_MODEL_FILENAME
            self.task = utils.TASK
            self.device = device
            self.sr = utils.SAMPLING_RATE
            self.mapping = utils.HF_PT_MAPPING

            try:
                # Initialize model and related components
                log.info("Starting PyTorch Inference service...")

                try:
                    self.pretrained_model_path = hf_hub_download(repo_id=self.pytorch_converted_model_source,
                                                                 filename=self.pytorch_converted_model_filename)
                    log.info(f"Whisper pretrained model downloaded to {self.pretrained_model_path}")

                except Exception as e:
                    log.info(f"Unable to download the PyTorch model: {str(e)} - switching to model from HF for conversion")
                    self.get_hf_model()

                self.model = self.set_pt_model()

                if quantization:
                    self.model = torch.quantization.quantize_dynamic(self.model,
                                                                {torch.nn.Linear},
                                                                dtype=torch.qint8)

                self.model = self.model.cpu()
                self.processor = self.create_processor()

                InferenceService._initialized = True
                log.info("PyTorch Inference service started with success!")

            except Exception as e:
                log.error(f"Error during PyTorch Inference service init: {str(e)}")
                raise

        torch.set_num_threads(num_threads)
        log.info(f"Number of threads set to {num_threads} for PyTorch calculations")

    def get_hf_model(self):
        """
        """
        try:
            merged_model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

            pt_model_name = os.path.basename(self.model_name) + ".pth"
            pt_dir_name = os.path.join("assets","pt_models")

            self.pretrained_model_path = os.path.join(pt_dir_name, pt_model_name)

            if not os.path.exists(pt_dir_name):
                os.makedirs(pt_dir_name)
                log.info(f"Directory {pt_dir_name} created and will be used to store PyTorch models")
            else:
                log.info(f"Directory {pt_dir_name} exists, using it to save PyTorch model")

            torch.save(merged_model.state_dict(), self.pretrained_model_path)
            log.info(f"HF model saved to {self.pretrained_model_path} in PyTorch format for conversion")

        except Exception as e:
            log.error(f"Error during HuggingFace model loading: {str(e)}")
            raise

        return 1

    def map_hf_to_pt(self,pretrained_weights):

        def rename_key(key):
            new_key = key
            for k, v in self.mapping:
                new_key = new_key.replace(k, v)
            return new_key

        # Rename the keys in the state_dict
        updated_weights = {rename_key(k): v for k, v in pretrained_weights.items()}
        updated_weights.pop('proj_out.weight', None)

        return updated_weights

    def set_pt_model(self):

        model = whisper.load_model("medium")
        log.info("Whisper base model loaded")

        pretrained_model = torch.load(self.pretrained_model_path)
        log.info(f"Whisper pretrained model loaded from {self.pretrained_model_path}")

        # Extract state_dict if the loaded model is not already a state_dict
        if hasattr(pretrained_model, "state_dict"):
            pretrained_weights = pretrained_model.state_dict()  # extract the state dict
        else:
            pretrained_weights = pretrained_model  # it's already a state_dict

        #######################################################################

        updated_weights = self.map_hf_to_pt(pretrained_weights)
        model.load_state_dict(updated_weights, strict=True)

        log.info(f"Model weights mapped from HuggingFace model to PyTorch")

        # Activate to save converted model and/or its weights
        # torch.save(model, 'src/model/whisper_pretrained_converted.pth')
        # torch.save(updated_weights, 'src/model/whisper_pretrained_converted_weights.pth')

        ######################################################################

        model.to(self.device)
        model.requires_grad_(False)
        model.eval()

        log.info("Whisper PyTorch model loaded on " + str(self.device))

        return model


    def create_processor(self):
        """
        """
        try:
            processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            log.info("WhisperProcessor created")
            return processor
        except Exception as e:
            log.error(f"Error during WhisperProcessor creation: {str(e)}")
            raise


    def preprocess_audio(self, waveform):
        """
        """
        # compute log-Mel input features from input audio array
        mel = self.processor.feature_extractor(waveform, sampling_rate=self.sr).input_features
        return torch.tensor(mel, dtype=torch.float32)


    def inference_pipeline(self,audio_path):

        log.info("1 - Starting audio load:")
        # waveform, sample_rate = librosa.load(audio_path, sr=self.sr)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)[0]

        log.info("2 - starting preprocessing:")
        audio_features = self.preprocess_audio(waveform)

        log.info("3 - Starting forward pass:")

        with torch.no_grad():
            result = whisper.decode(
                self.model,
                audio_features,
                options=whisper.DecodingOptions(
                    fp16=False,
                    language="it",
                    without_timestamps=True,
                    suppress_blank=False,
                    suppress_tokens=[],
                ),
            )

        return result[0].text


    async def transcribe(self, audio_path: str) -> DecodingResult | list[DecodingResult]:
        """
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Transcribing the following file audio: {audio_path}")
            log.info("Transcription started...")

            text = await loop.run_in_executor(
                None,
                lambda: self.inference_pipeline(audio_path)
                )

            log.info("Transcription completed!")

            return text

        except Exception as e:
            log.error(f"Error during transcription: {str(e)}")
            raise
