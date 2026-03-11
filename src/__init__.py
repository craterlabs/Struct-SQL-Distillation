"""
Struct-SQL Distillation source package.

Modules:
    config_loader           - INI-file configuration management
    llm_initializer         - Factory functions for Azure OpenAI and local HF models
    database_manager        - SQLite query execution and semantic schema generation
    bird_schema_loader      - BIRD benchmark schema metadata loader
    distillation_utils      - LoRA training core (SFTTrainer-based)
    distillation_data_utils - Dataset baking for knowledge distillation
    data_processing/        - SQL classification, curation pipeline, and reporting
"""
