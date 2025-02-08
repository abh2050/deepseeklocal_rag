# DeepSeek Local Inference with Ollama for Apple Silicon

A production-ready implementation for running DeepSeek language models locally on Apple Silicon, leveraging Metal acceleration through Ollama. This project implements a RAG (Retrieval-Augmented Generation) pipeline for document analysis with vector-based semantic search.

## Overview

This system enables local LLM inference using DeepSeek models, optimized for Apple Silicon architecture. It includes a RAG implementation using ChromaDB as the vector store, allowing for context-aware document analysis without external API dependencies.

## Key Features
- **Local Inference**: Full local execution on Apple Silicon using Metal acceleration
- **RAG Pipeline**: Document processing with semantic search via ChromaDB
- **Streaming Inference**: Real-time token streaming with visible reasoning steps
- **Resource Management**: Automatic memory cleanup and vector store persistence
- **Metal Optimization**: Leverages Apple's Metal API for GPU acceleration

## System Requirements

- Apple Silicon (M1/M2/M3) Mac
- macOS 13.0+ (Ventura or later)
- Python 3.9+
- 16GB+ RAM recommended

## Quick Start

1. Install Ollama:
```bash
brew install ollama
```

2. Pull the DeepSeek model:
```bash
ollama pull deepseek-coder:7b-instruct-q4_K_M
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run app.py
```

## Architecture

### Document Processing Pipeline
1. PDF ingestion via PyMuPDF
2. Chunk segmentation using recursive character splitting
3. Vector embedding generation
4. ChromaDB persistence layer

### Inference Pipeline
1. Query embedding generation
2. Semantic search in vector space
3. Context assembly and prompt construction
4. Local inference via Ollama
5. Stream processing and response rendering

## Configuration

Key parameters can be adjusted in `config.py`:
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DISTANCE_THRESHOLD = 0.75
MAX_CONTEXT_CHUNKS = 4
```

## Performance Considerations

- **Memory Usage**: Vector store size scales with document corpus
- **Inference Speed**: Dependent on chosen model quantization
- **Disk Usage**: ~4GB for base model, additional for vector store

## Troubleshooting

### Common Issues

1. **OOM Errors**
   - Reduce `MAX_CONTEXT_CHUNKS`
   - Use model quantization

2. **Slow Inference**
   - Verify Metal acceleration
   - Monitor thermal throttling
   - Consider lighter model variants

3. **Vector Store Issues**
   - Clear persistent storage
   - Rebuild index

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

### Testing
```bash
pytest tests/
```

## License
MIT License - See LICENSE file for details

## Acknowledgments
- Ollama Team for Metal optimization
- ChromaDB for vector store implementation
