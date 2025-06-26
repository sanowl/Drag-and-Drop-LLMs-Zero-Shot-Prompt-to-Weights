"""
Text encoders for extracting semantic embeddings from prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import logging

logger = logging.getLogger(__name__)


class SentenceBERTEncoder(nn.Module):
    """
    Sentence-BERT encoder as used in the paper (all-MiniLM-L6-v2)
    Section 2.4: Prompt Embedding
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = 384  # Exact dimension from paper
        
        # Freeze encoder as mentioned in paper
        for param in self.model.parameters():
            param.requires_grad = False
            
        logger.info(f"Initialized Sentence-BERT encoder: {model_name}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling as described in Sentence-BERT paper"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, prompts: List[str]) -> torch.Tensor:
        """
        Extract embeddings from prompts
        Args:
            prompts: List of text prompts (batch of prompts as in paper)
        Returns:
            Embeddings tensor [batch_size, 384]
        """
        # Handle sequence length as mentioned in paper (512 max for BERT)
        encoded_input = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=512
        ).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1) 