
#!/usr/bin/env python3
"""
Simplified PDF RAG System

This script combines vector search with Graph RAG concepts to answer questions
about the PDF content. It uses both direct vector similarity and relationship
expansion to find relevant information.
"""

import os
import json
from pymilvus import MilvusClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 

# Initialize components (will be initialized in main)
milvus_client = None
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Collection names
entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"

def load_extracted_data():
    """Load the extracted data from the PDF processing."""
    with open("extracted_data.json", 'r') as f:
        return json.load(f)

def search_relations(query: str, top_k: int = 10):
    """Search for relevant relations using vector similarity."""
    query_embedding = embedding_model.embed_query(query)
    
    relation_search_res = milvus_client.search(
        collection_name=relation_col_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "text"],
    )[0]
    
    return [res['entity']['text'] for res in relation_search_res]

def search_passages(query: str, top_k: int = 5):
    """Search for relevant passages using vector similarity."""
    query_embedding = embedding_model.embed_query(query)
    
    passage_search_res = milvus_client.search(
        collection_name=passage_col_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "text"],
    )[0]
    
    return [res['entity']['text'] for res in passage_search_res]

def search_entities(query: str, top_k: int = 5):
    """Search for relevant entities using vector similarity."""
    query_embedding = embedding_model.embed_query(query)
    
    entity_search_res = milvus_client.search(
        collection_name=entity_col_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "text"],
    )[0]
    
    return [res['entity']['text'] for res in entity_search_res]

def combine_context(relations: list, passages: list, entities: list):
    """Combine different types of context for comprehensive answers."""
    context_parts = []
    
    if relations:
        context_parts.append("RELEVANT RELATIONSHIPS:")
        for i, relation in enumerate(relations[:5], 1):
            context_parts.append(f"{i}. {relation}")
    
    if passages:
        context_parts.append("\nRELEVANT PASSAGES:")
        for i, passage in enumerate(passages[:3], 1):
            context_parts.append(f"{i}. {passage[:300]}...")
    
    if entities:
        context_parts.append("\nRELEVANT ENTITIES:")
        for i, entity in enumerate(entities[:5], 1):
            context_parts.append(f"{i}. {entity}")
    
    return "\n".join(context_parts)

def generate_answer(query: str, context: str):
    """Generate answer using the combined context."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an expert assistant analyzing financial and investment documents. 
Use the following context to answer the question accurately and comprehensively.

Question: {question}

Context:
{context}

Instructions:
- Provide specific, factual answers based on the context
- If specific numbers or data are mentioned, include them
- If the context doesn't contain enough information, say so
- Be concise but complete

Answer:""")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    try:
        answer = rag_chain.invoke({"question": query, "context": context})
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

def answer_question(query: str):
    """Answer a question using combined vector search and Graph RAG concepts."""
    
    print(f"Query: {query}")
    print("-" * 60)
    
    # Search across all collections
    relations = search_relations(query, top_k=8)
    passages = search_passages(query, top_k=4)
    entities = search_entities(query, top_k=5)
    
    print(f"Found {len(relations)} relevant relations, {len(passages)} passages, {len(entities)} entities")
    
    # Combine context
    context = combine_context(relations, passages, entities)
    
    # Generate answer
    answer = generate_answer(query, context)
    
    print(f"\nAnswer: {answer}")
    
    return {
        "query": query,
        "relations": relations,
        "passages": passages,
        "entities": entities,
        "answer": answer
    }

def main():
    """Main function for interactive querying."""
    
    global milvus_client
    
    # Initialize Milvus client
    milvus_client = MilvusClient(uri="./milvus.db")
    
    # Load data to verify system is ready
    data = load_extracted_data()
    print("PDF RAG System Ready!")
    print(f"Loaded {len(data['entities'])} entities, {len(data['relations'])} relations, and {len(data['passages'])} passages")
    print("\n" + "="*60)
    
    # Example queries
    example_queries = [
        "What are the transaction costs for divestment from thermal coal?",
        "What legislative mandates affect CalPERS investments?",
        "What is the impact of divestment on fund performance?",
        "What are the ESG considerations for emerging markets investments?",
        "What are the costs for divesting from Sudan and Iran?"
    ]
    
    print("Example queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "="*60)
    
    # Test with a specific query first
    print("Testing with example query...")
    result = answer_question("What are the transaction costs for divestment from thermal coal?")
    
    print("\n" + "="*60)
    print("System is ready for interactive use!")
    print("You can now ask questions about your PDF content.")
    
    # Interactive loop
    while True:
        try:
            query = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            answer_question(query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
