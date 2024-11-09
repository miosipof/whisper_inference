
MODEL_NAME = "miosipof/asr_EN_medium_v1"
MERGED_MODEL_NAME = "miosipof/asr_EN_medium_v1_merged"
PRETRAINED_MODEL_PTH = ""
PRETRAINED_MODEL_FILENAME = "whisper_pretrained_converted.pth"
TASK = "transcribe"
SAMPLING_RATE = 16000

OV_MODEL = "miosipof/asr_openvino_int4"

HF_PT_MAPPING = [('model.', ''),
           ('decoder.layers', 'decoder.blocks'),
           ('encoder.layers', 'encoder.blocks'),

           ('encoder.embed_positions.weight', 'encoder.positional_embedding'),

           ('self_attn.k_proj', 'attn.key'),
           ('self_attn.q_proj', 'attn.query'),
           ('self_attn.v_proj', 'attn.value'),
           ('self_attn.out_proj', 'attn.out'),

           ('self_attn_layer_norm', 'attn_ln'),
           ('final_layer_norm', 'mlp_ln'),
           ('fc1', 'mlp.0'),
           ('fc2', 'mlp.2'),

           ('encoder_attn.k_proj', 'cross_attn.key'),
           ('encoder_attn.v_proj', 'cross_attn.value'),
           ('encoder_attn.q_proj', 'cross_attn.query'),
           ('encoder_attn.out_proj', 'cross_attn.out'),
           ('encoder_attn_layer_norm', 'cross_attn_ln'),

           ('decoder.embed_positions.weight', 'decoder.positional_embedding'),
           ('decoder.embed_tokens', 'decoder.token_embedding'),

           ('encoder.layer_norm', 'encoder.ln_post'),

           ('decoder.layer_norm', 'decoder.ln'),
           ]