#!/usr/bin/env python3
"""
PDF to Graph RAG Ingestion Script

This script extracts text from PDF files, identifies entities and relations,
and ingests them into Milvus for Graph RAG processing.

Usage:
    python pdf_to_graphrag.py --pdf_path item09a-01_a.pdf
"""

import os
import argparse
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import pdfplumber
import spacy
from pymilvus import MilvusClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm
import json

# Set OpenAI API key (use the same key from your working script)
os.environ["OPENAI_API_KEY"] =

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install the English spaCy model: python -m spacy download en_core_web_sm")
    exit(1)

class PDFToGraphRAG:
    def __init__(self, milvus_uri: str = "./milvus.db"):
        """Initialize the PDF to Graph RAG processor."""
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Collections
        self.entity_col_name = "entity_collection"
        self.relation_col_name = "relation_collection"
        self.passage_col_name = "passage_collection"
        
        # Data storage
        self.entities = []
        self.relations = []
        self.passages = []
        self.entityid_2_relationids = defaultdict(list)
        self.relationid_2_passageids = defaultdict(list)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        print(f"Extracting text from {pdf_path}...")
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        print(f"Extracted {len(text)} characters from {len(pdf.pages)} pages")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk.strip()]
    
    def extract_entities_and_relations(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract entities and relations from text using spaCy and LLM."""
        print("Extracting entities and relations...")
        
        # Use spaCy for initial entity extraction
        doc = nlp(text)
        
        # Extract named entities
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                entities.add(ent.text.strip())
        
        # Use LLM to extract more complex relations
        relations = self.extract_relations_with_llm(text)
        
        return list(entities), relations
    
    def extract_relations_with_llm(self, text: str) -> List[str]:
        """Use LLM to extract relations from text."""
        # Split text into smaller chunks for LLM processing
        chunks = self.chunk_text(text, chunk_size=2000, overlap=100)
        all_relations = []
        
        prompt_template = """
        Extract factual relationships from the following text. 
        Return a JSON list of relationships in the format: [["subject", "predicate", "object"], ...]
        
        Focus on:
        - Who did what to whom
        - What happened when/where
        - Cause and effect relationships
        - Definitions and descriptions
        - Hierarchical relationships
        
        Text: {text}
        
        Return only the JSON array, no other text.
        """
        
        for chunk in tqdm(chunks, desc="Extracting relations with LLM"):
            try:
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | self.llm | JsonOutputParser()
                
                result = chain.invoke({"text": chunk})
                
                if isinstance(result, list):
                    for relation in result:
                        if isinstance(relation, list) and len(relation) == 3:
                            # Convert to string format like the original system
                            relation_str = f"{relation[0]} {relation[1]} {relation[2]}"
                            all_relations.append(relation_str)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        return list(set(all_relations))  # Remove duplicates
    
    def create_milvus_collections(self):
        """Create Milvus collections for entities, relations, and passages."""
        embedding_dim = len(self.embedding_model.embed_query("test"))
        
        collections = [self.entity_col_name, self.relation_col_name, self.passage_col_name]
        
        for collection_name in collections:
            if self.milvus_client.has_collection(collection_name=collection_name):
                print(f"Dropping existing collection: {collection_name}")
                self.milvus_client.drop_collection(collection_name=collection_name)
            
            print(f"Creating collection: {collection_name}")
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
            )
    
    def insert_data_to_milvus(self, text_chunks: List[str]):
        """Insert entities, relations, and passages into Milvus."""
        print("Inserting data into Milvus...")
        
        # Insert entities
        self.milvus_insert(self.entity_col_name, self.entities)
        
        # Insert relations
        self.milvus_insert(self.relation_col_name, self.relations)
        
        # Insert passages
        self.milvus_insert(self.passage_col_name, text_chunks)
        
        print("Data insertion completed!")
    
    def milvus_insert(self, collection_name: str, text_list: List[str], batch_size: int = 100):
        """Insert text data into Milvus collection with embeddings."""
        for row_id in tqdm(range(0, len(text_list), batch_size), desc=f"Inserting {collection_name}"):
            batch_texts = text_list[row_id : row_id + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            
            batch_ids = [row_id + j for j in range(len(batch_texts))]
            batch_data = [
                {
                    "id": id_,
                    "text": text,
                    "vector": vector,
                }
                for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
            ]
            
            self.milvus_client.insert(
                collection_name=collection_name,
                data=batch_data,
            )
    
    def build_entity_relation_mappings(self):
        """Build mappings between entities, relations, and passages."""
        print("Building entity-relation mappings...")
        
        # Build entity to relation mapping
        for relation_id, relation in enumerate(self.relations):
            # Extract entities from relation string
            relation_parts = relation.split()
            if len(relation_parts) >= 3:
                subject = relation_parts[0]
                object_entity = relation_parts[-1]
                
                # Find entity indices
                if subject in self.entities:
                    subject_id = self.entities.index(subject)
                    self.entityid_2_relationids[subject_id].append(relation_id)
                
                if object_entity in self.entities:
                    object_id = self.entities.index(object_entity)
                    self.entityid_2_relationids[object_id].append(relation_id)
        
        # Build relation to passage mapping (simplified - all relations map to all passages)
        for relation_id in range(len(self.relations)):
            for passage_id in range(len(self.passages)):
                self.relationid_2_passageids[relation_id].append(passage_id)
    
    def process_pdf(self, pdf_path: str):
        """Main method to process PDF and ingest into Milvus."""
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk text into passages
        self.passages = self.chunk_text(text, chunk_size=1000, overlap=200)
        print(f"Created {len(self.passages)} text chunks")
        
        # Step 3: Extract entities and relations
        self.entities, self.relations = self.extract_entities_and_relations(text)
        print(f"Extracted {len(self.entities)} entities and {len(self.relations)} relations")
        
        # Step 4: Build mappings
        self.build_entity_relation_mappings()
        
        # Step 5: Create Milvus collections
        self.create_milvus_collections()
        
        # Step 6: Insert data into Milvus
        self.insert_data_to_milvus(self.passages)
        
        print("PDF processing completed successfully!")
        print(f"Summary:")
        print(f"  - Entities: {len(self.entities)}")
        print(f"  - Relations: {len(self.relations)}")
        print(f"  - Passages: {len(self.passages)}")
        
        return {
            "entities": self.entities,
            "relations": self.relations,
            "passages": self.passages,
            "entityid_2_relationids": dict(self.entityid_2_relationids),
            "relationid_2_passageids": dict(self.relationid_2_passageids)
        }

def main():
    parser = argparse.ArgumentParser(description="Process PDF and ingest into Milvus for Graph RAG")
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file")
    parser.add_argument("--milvus_uri", default="./milvus.db", help="Milvus database URI")
    parser.add_argument("--output", help="Output file to save extracted data (JSON)")
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return
    
    # Initialize processor
    processor = PDFToGraphRAG(milvus_uri=args.milvus_uri)
    
    # Process PDF
    try:
        result = processor.process_pdf(args.pdf_path)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    main()
