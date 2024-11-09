import os
from transformers import WhisperProcessor, logging as transformers_log
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import torchaudio
import torch
import numpy as np
import time

from src import log
from src.utils import utils

import asyncio



class OpenVinoService:
    _initialized = False

    def __init__(self, language='it'):
        if not OpenVinoService._initialized:
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            transformers_log.set_verbosity_error()
            self.model_name = utils.MERGED_MODEL_NAME
            self.ov_model_name = utils.OV_MODEL
            self.language = language
            self.task = utils.TASK
            self.device = "CPU"
            self.sr = utils.SAMPLING_RATE

            try:
                # Initialize model and related components
                log.info("Starting OpenVino service...")
                self.model = self.get_openvino_model()
                self.compile_openvino_model()
                self.processor = self.create_processor()

                OpenVinoService._initialized = True
                log.info("OpenVino service started with success!")
            except Exception as e:
                log.error(f"Error during OpenVino service init: {str(e)}")
                raise

    def get_openvino_model(self):
        """
        """
        ov_config = {"CACHE_DIR": ""}
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(self.ov_model_name, ov_config=ov_config, compile=False)
        log.info("OpenVino model loaded from " + str(self.ov_model_name))

        # try:[openvino,nncf]optimum-cli export openvino --model miosipof/asr_double_training_15-10-2024_merged --weight-format int4 asr_openvino_int4
        #     ov_model_path = Path("src/model/" + self.model_name.replace("/", "_"))
        #     ov_config = {"CACHE_DIR": ""}
        #
        #     if not ov_model_path.exists():
        #         self.model = OVModelForSpeechSeq2Seq.from_pretrained(
        #             self.model_name,
        #             ov_config=ov_config,
        #             export=True,
        #             compile=False,
        #             load_in_8bit=False,
        #         )
        #         self.model.half()
        #         self.model.save_pretrained(ov_model_path)
        #         log.info("HF model converted to OpenVino and saved in " + str(ov_model_path))
        #     else:
        #         self.model = OVModelForSpeechSeq2Seq.from_pretrained(ov_model_path, ov_config=ov_config, compile=False)
        #         log.info("OpenVino model loaded from " + str(ov_model_path))
        #
        # except Exception as e:
        #     log.error(f"Error during OpenVino model loading: {str(e)}")
        #     raise

        return self.model


    def compile_openvino_model(self):
        """
        """
        try:

            if torch.cuda.is_available():
                log.info("Model loaded on GPU")
                self.device = "GPU"
            else:
                log.info("Model loaded on CPU")
                self.device = "CPU"

            self.model.to(self.device)
            self.model.compile()

            log.info("OpenVino model compiled successfully")

        except Exception as e:
            log.error(f"Error during OpenVino model compilation: {str(e)}")
            raise

        return self.model


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
        audio_features = self.processor.feature_extractor(waveform, sampling_rate=self.sr).input_features[0]
        audio_features = torch.tensor(np.array([audio_features]))

        return audio_features


    def openvino_pipeline(self,audio_path):

        print("1 - starting audio load:", time.time())
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)[0]
        print("2 - starting preprocessing:", time.time())
        audio_features = self.preprocess_audio(waveform)

        print("3 - starting forward pass:", time.time())
        predicted_ids = self.model.generate(audio_features, max_new_tokens=224)

        print("4 - starting decoding:", time.time())
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0]


    async def transcribe(self, audio_path: str) -> str:
        """
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Transcribing the following file audio: {audio_path}")

            print("0 - starting the loop:",time.time())
            text = await loop.run_in_executor(
                None,
                lambda: self.openvino_pipeline(audio_path)
                )

            print("5 - all done:", time.time())
            log.info("Transcription completed!")
            return text

        except Exception as e:
            log.error(f"Error during transcription: {str(e)}")
            raise
