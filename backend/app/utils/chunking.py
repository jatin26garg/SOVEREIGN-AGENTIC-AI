import re
from typing import List,Dict,Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text :str,chunk_size:int = 1000,chunk_overlap:int = 150)->List[str]:

    text = re.sub(r'\s+',' ',text).strip()

    if not text:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    ) 
    chunks = splitter.split_text(text)
    cleaned_chunks = []

    for chunk in chunks:
        chunk = chunk.strip()

        if chunk and chunk[-1] not in['.', '!','?']:
            last_period = chunk.rfind('.')
            if last_period > len(chunk)//2 :
                chunk = chunk[:last_period+1]

        if chunk:
            cleaned_chunks.append(chunk)
    return cleaned_chunks

def chunk_document(text:str, metadata:Dict[str, Any],chunk_size:int =500,chunk_overlap:int = 50, )->List[Dict[str,Any]]:
    chunks = chunk_text(text,chunk_size,chunk_overlap)

    return[
        {
            "content" : chunk,
            "metadata" : {
                **metadata,
                "chunk_index" : i,
                "total_chunks" : len(chunks),
                "chunk_size" : len(chunk)
            }
        }
        for i, chunk in enumerate(chunks)
    ]

