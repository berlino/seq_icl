optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "src.models.sequence.SequenceModel",
    "lm": "src.models.sequence.long_conv_lm.ConvLMHeadModel",
    "lm_simple": "src.models.sequence.simple_lm.SimpleLMHeadModel",
    "lm_simple_noffn": "src.models.sequence.simple_lm.SimpleLMHeadModelNoFFN",
    "hybrid_lm": "src.models.sequence.simple_lm.HybridLMHeadModel",
    "vit_b_16": "src.models.baselines.vit_all.vit_base_patch16_224",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "ff": "src.models.sequence.ff.FF",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "mha-rope": "src.models.sequence.mha_rope.MultiHeadAttentionWithRope",
    "s4d": "src.models.sequence.ssm.s4d.S4D",
    "s4_simple": "src.models.sequence.ssm.s4_simple.SimpleS4Wrapper",
    "long-conv": "src.models.sequence.long_conv.LongConv",
    "h3": "src.models.sequence.h3.H3",
    "h3-conv": "src.models.sequence.h3_conv.H3Conv",
    "hyena": "src.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src.models.sequence.hyena.HyenaFilter",
    "vit": "src.models.sequence.mha.VitAttention",
    "gilr": "src.models.sequence.rnn.gilr.GILRLayer",
    "lru": "src.models.sequence.rnn.lru.LRULayer",
    "lstm": "src.models.sequence.rnn.lstm.LSTMLayer",
    "rwkv": "src.models.sequence.rnn.rwkv.RWKVLayer",
    "retention": "src.models.sequence.rnn.retention.MultiScaleRetention",
    "retentionv2": "src.models.sequence.rnn.retentionv2.MultiScaleRetention",
    "mamba": "src.models.sequence.rnn.mamba.MambaLayer",
    "gla": "src.models.sequence.rnn.gla.GatedLinearAttention",
    "ngram": "src.models.sequence.rnn.ngram.Ngram",
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
}
