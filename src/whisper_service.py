import asyncio

from huggingface_hub import login
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    logging as transformers_log,
    PreTrainedModel
)
from peft import PeftModel, PeftConfig
from src import log
from src.utils import utils

import torch
import os

class WhisperService:
    _initialized = False

    def __init__(self, language='it'):
        if not WhisperService._initialized:
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            transformers_log.set_verbosity_error()
            self.model_name = utils.MODEL_NAME
            self.language = language
            self.task = utils.TASK

            try:
                # Initialize model and related components
                log.info("Starting Whisper service...")
                self.peft_config = self.generate_model_config()
                self.model = self.get_whisper_model_from_hf(self.peft_config)
                self.tokenizer = self.create_tokenizer(self.peft_config)
                self.processor = self.create_processor(self.peft_config)
                self.pipeline_asr, self.forced_decoder_ids = self.create_whisper_pipeline(
                    self.model, self.tokenizer, self.processor
                )
                WhisperService._initialized = True
                log.info("Whisper service started with success!")
            except Exception as e:
                log.error(f"Error during Whisper service init: {str(e)}")
                raise

    def generate_model_config(self) -> PeftConfig:
        """
        """
        try:
            login(token=os.environ['API_TOKEN'])
            config = PeftConfig.from_pretrained(self.model_name)
            log.info("Model config generated")
            return config
        except Exception as e:
            log.error(f"Error during model config generation: {str(e)}")
            raise

    def get_whisper_model_from_hf(self, peft_config: PeftConfig) -> PeftModel:
        """
        """
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                    peft_config.base_model_name_or_path
                )
            # Check if GPU is available
            if torch.cuda.is_available():
                log.info("Model loaded on GPU")
            else:
                log.info("Model loaded on CPU")

            model = PeftModel.from_pretrained(model, self.model_name)
            log.info("Whisper model configured with PeftModel")
            return model
        except Exception as e:
            log.error(f"Error during Whisper model loading: {str(e)}")
            raise

    def create_processor(self, peft_config: PeftConfig) -> WhisperProcessor:
        """
        """
        try:
            processor = WhisperProcessor.from_pretrained(
                peft_config.base_model_name_or_path,
                language=self.language,
                task=self.task
            )
            log.info("WhisperProcessor created")
            return processor
        except Exception as e:
            log.error(f"Error during WhisperProcessor creation: {str(e)}")
            raise

    def create_tokenizer(self, peft_config: PeftConfig) -> WhisperTokenizer:
        """
        """
        try:
            tokenizer = WhisperTokenizer.from_pretrained(
                peft_config.base_model_name_or_path,
                language=self.language,
                task=self.task
            )
            log.info("WhisperTokenizer created")
            return tokenizer
        except Exception as e:
            log.error(f"Error during WhisperTokenizer creation: {str(e)}")
            raise

    def create_whisper_pipeline(self, model: PreTrainedModel, tokenizer: WhisperTokenizer,
                                processor: WhisperProcessor) -> tuple:
        """
        """
        try:
            feature_extractor = processor.feature_extractor
            pipe_lora = AutomaticSpeechRecognitionPipeline(
                model=model,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor
            )
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            log.info("Pipeline created")
            return pipe_lora, forced_decoder_ids
        except Exception as e:
            log.error(f"Error during Pipeline creation: {str(e)}")
            raise

    async def transcribe(self, audio_path: str) -> str:
        """
        """
        try:
            loop = asyncio.get_event_loop()
            log.info(f"Transcribing the following file audio: {audio_path}")
            with torch.cuda.amp.autocast():
                text = await loop.run_in_executor(
                    None,
                    lambda:
                    self.pipeline_asr(audio_path, generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids},
                                      max_new_tokens=255)["text"]
                )
            log.info("Transcription completed!")
            return text
        except Exception as e:
            log.error(f"Error during transcription: {str(e)}")
            raise
