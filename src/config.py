
import torch


class Config:

    # Vocabulary
    SRC_VOCAB_SIZE = 24_000
    TGT_VOCAB_SIZE = 24_000

    PAD_ID = 0
    BOS_ID = 2
    EOS_ID = 3

    # Transformer architecture
    D_MODEL = 256
    NUM_HEADS = 4

    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4

    D_FF = 1024
    MAX_SEQ_LEN = 256
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    EPOCHS = 40

    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

    NUM_WORKERS = 0
    SEED = 42

    # Device
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
