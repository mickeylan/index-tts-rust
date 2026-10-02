# Third Party Notices

This project includes or references the following third-party software components.

## IndexTTS-2.5

**Component**: IndexTTS-2.5 Text-to-Speech Model  
**Source**: https://github.com/index-tts/index-tts  
**License**: Bilibili Model Use License Agreement  

> **Important**: The IndexTTS-2.5 model weights and inference code are governed by a separate license agreement. Users must:
> - Obtain authorization for cloned voices
> - Not use for impersonation, fraud, privacy infringement, or other unlawful purposes
> - Comply with the model license and applicable law

## Hugging Face Transformers

**Component**: Hugging Face Transformers (via compatibility layer)  
**Source**: https://github.com/huggingface/transformers  
**License**: Apache License 2.0  

Copyright 2018-2024 The Hugging Face Team. All rights reserved.

## Candle

**Component**: Candle (ML framework)  
**Source**: https://github.com/huggingface/candle  
**License**: Apache License 2.0  

Copyright 2023-2024 Hugging Face Inc.

## ONNX Runtime

**Component**: ONNX Runtime  
**Source**: https://onnxruntime.ai/  
**License**: MIT License  

Copyright (c) Microsoft Corporation.

## Safetensors

**Component**: Safetensors (safe tensor format)  
**Source**: https://github.com/huggingface/safetensors  
**License**: Apache License 2.0  

Copyright 2021-2024 Hugging Face Inc.

## Hound

**Component**: Hound (WAV I/O)  
**Source**: https://github.com/ruuda/hound  
**License**: (BSD-2-Clause or Apache 2.0)  

## Symphonia

**Component**: Symphonia (audio decoding)  
**Source**: https://github.com/padenot/symphonia  
**License**: MPL-2.0  

Copyright 2015-2024 the Contributors.

## Rubato

**Component**: Rubato (audio resampling)  
**Source**: https://github.com/HEnquist/rubato  
**License**: MIT License  

Copyright (c) 2019-2024 Håkon Endrestol.

## Other Dependencies

| Package | License |
|---------|---------|
| rand | MIT OR Apache-2.0 |
| serde | MIT OR Apache-2.0 |
| thiserror | MIT OR Apache-2.0 |
| tokio | MIT OR Apache-2.0 |
| clap | MIT OR Apache-2.0 |
| tracing | MIT OR Apache-2.0 |

## Licenses Summary

- **Apache 2.0**: Candle, Safetensors, ONNX Runtime, Hound (some), Transformers
- **MIT**: ONNX Runtime, Rubato, Rand, Serde, Thiserror, Tokio, Clap, Tracing
- **MPL 2.0**: Symphonia
- **Proprietary**: IndexTTS-2.5 model weights (separate agreement required)

## Model Weights

This repository does **not** include IndexTTS-2.5 model weights. Users must:

1. Obtain the official IndexTTS-2.5 model weights from the official source
2. Agree to the Bilibili Model Use License Agreement
3. Keep the supplied license and disclaimer

See: https://github.com/index-tts/index-tts for model downloads and licensing.

## Voice Cloning Ethics

This software enables voice cloning, which raises ethical concerns:

- **Always obtain explicit consent** before cloning someone's voice
- **Never impersonate** individuals without authorization
- **Do not use** for fraud, defamation, or harassment
- **Comply with** local laws regarding voice synthesis and biometric data

The authors of this project are not responsible for misuse of this software.
