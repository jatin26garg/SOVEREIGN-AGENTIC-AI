import os
import uuid
from typing import List, Dict,Any , Tuple, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient,models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.utils.chunking import chunk_document
from app.utils.file_handlers import extract_text_from_file

from FlagEmbedding import BGEM3FlagModel


class RAGService:
    
    def __init__(self):

        
        self.client = QdrantClient(
                            host= settings.QDRANT_HOST,
                            port= settings.QDRANT_PORT,
                            timeout=60.0,
                            )
        
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        

        self.embeddings = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=False,
            device="cpu"
        )
        print(f" connecting to ollama model")
        self.llm = ChatOllama(
            model= settings.CHAT_MODEL,
            base_url= settings.OLLAMA_BASE_URL,
            temperature = 0.3,
            disable_streaming=False,
        )
        print(f" qwen model is connected")
        
        self.prompt = ChatPromptTemplate.from_template("""
                    You are a helpful assistant that answers questions based on the provided context.

    IMPORTANT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, say "I don't have information about that."
3. DO NOT make up information.
4. Each source below includes a filename in brackets like [File: name.pdf]. 
   Use these filenames to answer questions about specific documents.
5. If the user asks about a specific file by name, mention that file's content 
   even if the filename doesn't appear in the text itself.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")
        self._documents = {}
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            
            print(f" creating Hybrid collectiion: {self.collection_name}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                
                vectors_config={
                    "dense" : models.VectorParams(
                            size=settings.DENSE_VECTOR_SIZE,
                            distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config ={
                    "sparse": models.SparseVectorParams(
                        index = models.SparseIndexParams(
                            on_disk=False,
                        ),
                        modifier=models.Modifier.IDF,
                    )
                }
                
            )
            print(f"Hybrid collection {self.collection_name} created !")
    
    def _get_embeddings(self, texts:List[str])-> Tuple[List[List[float]] , List[Dict]]:
        
        output = self.embeddings.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        dense_embeddings = output['dense_vecs']
        sparse_embeddings = []
        for sparse_vec  in  output['lexical_weights']:
            indices = list(sparse_vec.keys())
            values  = list(sparse_vec.values())
            sparse_embeddings.append({
                "indices" : indices,
                "values" : values,
            })
            
            
        return dense_embeddings,sparse_embeddings
    
    def _get_query_embeddings(self, query:str)->Tuple[List[float], Dict]:
        
        output = self.embeddings.encode(
            query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sparse_query = output['lexical_weights']
        
        dense_query= output['dense_vecs']
        
        sparse_query_formated={
            "indices": list(sparse_query.keys()),
            "values": list(sparse_query.values()),
        }
        return dense_query, sparse_query_formated
        
    def _reciprocal_rank_fusion(self,
                                dense_results:List[models.ScoredPoint],
                                sparse_results:List[models.ScoredPoint], 
                                k:int =60,
                                dense_weight:float= 0.5,
                                sparse_weight:float = 0.5,
                                )->List[Tuple[str,Any]]:
        
        dense_dict = {point.id: point for point in dense_results}
        sparse_dict = {point.id: point for point in sparse_results}
        
        
        all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
        
        combined_results = []
        
        for chunk_id in all_ids:
            rrf_score = 0
            dense_rank = None
            sparse_rank = None
            
            if chunk_id in dense_dict:
                for i,point in enumerate(dense_results,1):
                    if point.id == chunk_id:
                        dense_rank = i
                        break
                rrf_score += dense_weight*(1/(k+dense_rank))
            
            if chunk_id in sparse_dict:
                for i,point in enumerate(sparse_results,1):
                    if point.id == chunk_id:
                        sparse_rank = i
                        break
                rrf_score += sparse_weight*(1/(k+sparse_rank))
            
            if chunk_id in dense_dict:
                payload = dense_dict[chunk_id].payload
            else:
                payload = sparse_dict[chunk_id].payload
        
            combined_results.append({
                "id" : chunk_id,
                "rrf_score": rrf_score,
                "dense_rank":dense_rank,
                "sparse_rank": sparse_rank,
                "payload" :payload,
            })
        
        combined_results.sort(key = lambda x:x["rrf_score"] , reverse=True)
        
        return combined_results
    
    def hybrid_search(self,
                      query: str,
                      top_k:int = 3,
                      dense_limit:int = 10,
                      sparse_limit:int = 10,
                      dense_weight:float = 0.5,
                      sparse_weight:float = 0.5,
                      )->List[Dict[str,Any]]:
        
        print(f" performing dense Search on {query}")
        
        dense_query, sparse_query = self._get_query_embeddings(query)
        
        print(f" running dense or Semantic Search")
        
        dense_search = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query,
            using="dense",
            limit=dense_limit,
            with_payload=True,
            with_vectors=False,
        )
        #  print(f"   Found {len(dense_search)} dense results")
        
        dense_results = dense_search.points
        
        print(f" Running sparse Search or keyword search")
        
        sparse_search = self.client.query_points(
            collection_name=self.collection_name,
            query=(
                models.SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"],
                )
            ),
            using="sparse",
            limit=sparse_limit,
            with_payload=True,
            with_vectors=False,
        )
        sparse_results = sparse_search.points
        # print(f"   Found {len(sparse_search)} sparse results")
        print(f"   Combining results with RRF...")
        
        fused_results = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        final_results = []
        
        for result in fused_results[:top_k]:
            payload = result["payload"]
            
            final_results.append({
                "id":result["id"],
                "content":payload.get("content",""),
                "file_name":payload.get("file_name","Unknown"),
                "document_id":payload.get("document_id",""),
                "chunk_index": payload.get("chunk_index", 0),
                "rrf_score": round(result["rrf_score"], 4),
                "dense_rank": result["dense_rank"],
                "sparse_rank": result["sparse_rank"],
            })
        print(f" Hybrid search returned {len(final_results)} results")
        
        return final_results
 
        
    def process_document(self, file_content: bytes, file_name: str) -> str:
    
        print(f"extracting text from  :{file_name}")
        
        text = extract_text_from_file(file_content, file_name)
        
        if not text or len(text.strip()) == 0:
            raise ValueError(f"No text could be extracted from {file_name}")
        
        print(f"  Extracted file has {len(text)} chars")
        
        print(f" creating chunks")
        
        chunks_with_metadata = chunk_document(
            text=text,
            metadata={
                "file_name": file_name,
                "uploadedAt": datetime.now().isoformat(),
            },
            chunk_size=1000,
            chunk_overlap=150
        )
        
        print(f" created {len(chunks_with_metadata)} chunks")
        
        doc_id = str(uuid.uuid4())
        
        # ============================================================
        # ✅ FIX #1: Enrich chunks with filename BEFORE embedding
        # ============================================================
        # Why: The filename lives in metadata, but the LLM only reads the
        # "content" text. By prefixing the filename into the content that
        # gets embedded, the filename becomes searchable and visible to
        # the LLM. This fixes "tell me about X.pdf" questions.
        # ============================================================
        enriched_chunks = [
            f"[File: {chunk['content']}"
            for chunk in chunks_with_metadata
        ]
        
        print(f" generating embeddings with BGE-M3")
        
        # Embed the ENRICHED text (filename + content), not just content
        dense_embeddings, sparse_embeddings = self._get_embeddings(enriched_chunks)
        
        print(f" generated {len(dense_embeddings)} embeddings , sparse embeddings")
        
        points = []
        
        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_id = str(uuid.uuid4())
            
            point = models.PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": models.SparseVector(
                        indices=sparse_embeddings[i]["indices"],
                        values=sparse_embeddings[i]["values"],
                    )
                },
                payload={
                    # ✅ Store the ENRICHED content (searchable + visible to LLM)
                    "content": enriched_chunks[i],
                    
                    # ✅ Keep the ORIGINAL content separately (for display/UI)
                    "original_content": chunk_data["content"],
                    
                    "file_name": file_name,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks_with_metadata),
                    "uploaded_at": datetime.now().isoformat(),
                }
            )
            points.append(point)
        
        print(f" uploading {len(points)} points to Qdrant")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f" uploaded complete")
        
        self._documents[doc_id] = {
            "id": doc_id,
            "file_name": file_name,
            "chunk_count": len(chunks_with_metadata),
            "uploaded_at": datetime.now().isoformat(),
        }
        
        return doc_id  

    # Phrases that signal the user wants the WHOLE document (every item,
    # a full list, a complete summary) rather than a narrow fact. Plain
    # top-k chunk retrieval is the wrong tool for these - if the doc has
    # more chunks than top_k, the answer will silently miss content, which
    # is exactly what was happening for "tell me all the questions in
    # this document" (top_k=3 out of ~10+ chunks).
    _EXHAUSTIVE_QUERY_PATTERNS = (
        "all the", "all of the", "every ", "each question", "list all",
        "list every", "entire document", "whole document", "full document",
        "complete list", "how many questions", "all questions",
        "summarize the document", "summarize this document",
        "summarise the document", "summarise this document",
    )

    def _is_exhaustive_query(self, question: str) -> bool:
        q = question.lower()
        return any(pattern in q for pattern in self._EXHAUSTIVE_QUERY_PATTERNS)

    def _latest_document_id(self) -> Optional[str]:
        if not self._documents:
            return None
        return max(
            self._documents.values(),
            key=lambda d: d.get("uploaded_at", ""),
        )["id"]

    def get_chunks_for_document(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return every chunk belonging to a document, in original order.

        This was previously called by RAGTool.get_document_summary() but
        didn't exist on this class at all - any code path that tried to
        read a full document (instead of a top-k slice) raised an
        AttributeError. Implemented here by scrolling Qdrant for every
        point with this document_id, same filter pattern as delete_document.

        If document_id is omitted, defaults to the most recently uploaded
        document.
        """
        if document_id is None:
            document_id = self._latest_document_id()
        if document_id is None:
            return []

        chunks: List[Dict[str, Any]] = []
        scroll_limit = 100
        offset = None

        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=scroll_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, offset = scroll_result
            for point in points:
                payload = point.payload or {}
                chunks.append({
                    "id": point.id,
                    "content": payload.get("content", ""),
                    "file_name": payload.get("file_name", "Unknown"),
                    "document_id": payload.get("document_id", document_id),
                    "chunk_index": payload.get("chunk_index", 0),
                })
            if offset is None:
                break

        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks

    def _answer_from_context(self, question: str, context_parts: List[str], source_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = "\n\n".join(context_parts)

        print(f" generating answer --")
        print(f"   Generating answer with {settings.CHAT_MODEL}...")

        chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "success": True,
            "answer": answer,
            "sources": source_info,
        }

    def query(self, question:str, top_k: int = 6)->Dict[str,Any]:
        
        if not self._documents:
            return{
                "answer"  : ("no documents have been been uploaded .. please upload the document first"),
                "sources" : [],
            }

        # "Tell me all the questions in this document" etc. needs the full
        # document, not a semantic top-k slice - a query like that can
        # match chunks that merely repeat the word "question" (e.g. the
        # instructions header) while missing the actual question bodies.
        if self._is_exhaustive_query(question):
            doc_id = self._latest_document_id()
            chunks = self.get_chunks_for_document(doc_id)

            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information in your documents.",
                    "sources": [],
                }

            context_parts = []
            source_info = []
            running_len = 0

            for i, chunk in enumerate(chunks, 1):
                content = " ".join(chunk["content"].split())
                if running_len + len(content) > settings.MAX_FULL_DOC_CONTEXT_CHARS:
                    break
                running_len += len(content)

                context_parts.append(f"[Source{i}] \n [File: {chunk['file_name']}]\n {content}")
                source_info.append({
                    "source_index": i,
                    "file_name": chunk["file_name"],
                    "rrf_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "chunk_id": chunk["id"],
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                })

            return self._answer_from_context(question, context_parts, source_info)

        results = self.hybrid_search(
            query=question,
            top_k=top_k,
            dense_limit=max(10, top_k * 2),
            sparse_limit=max(10, top_k * 2),
            dense_weight=0.5,
            sparse_weight=0.5,
        )
        if not results:
            return {
                "answer": "I couldn't find any relevant information in your documents.",
                "sources": [],
            }
        context_parts  = []
        source_info  = []

        for i,result in enumerate(results,1):

            content = result["content"]

            content = " ".join(content.split())

            context_parts.append(f"[Source{i}] {content}")

            source_info.append({
                "source_index" : i,
                "file_name" : result["file_name"],
                "rrf_score" : result["rrf_score"],
                "dense_rank":result["dense_rank"],
                "sparse_rank":result["sparse_rank"],
                "chunk_id":result["id"],
                "content_preview"  :content[:200] + "..." if len(content) > 200 else content,
            })

        return self._answer_from_context(question, context_parts, source_info)

    def get_documents(self)->List[Dict[str,Any]]:
        return list(self._documents.values())
    
    def delete_document(self,doc_id :str)->bool:
        points_to_delete =[]
        scroll_limit = 100
        offset = None

        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=scroll_limit,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, offset = scroll_result
            if not points:
                break
            points_to_delete.extend([p.id for p in points])

            if offset is None:
                break
        
        if not points_to_delete:
            if doc_id in self._documents:
                del self._documents[doc_id]
            return False
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=points_to_delete),
        )
        
        if doc_id in self._documents:
            del self._documents[doc_id]
        
        print(f" document deleted finally")
        
        return True
    
_rag_service_instance: Optional["RAGService"] = None
        
def get_rag_service() -> "RAGService":
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance