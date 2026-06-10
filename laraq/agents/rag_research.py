"""RAG research agent using embeddings for retrieval."""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.embeddings import EmbeddingClient
from laraq.llm import LLMClient


class RAGResearchAgent(BaseAgent):
    """Research agent that uses RAG (Retrieval Augmented Generation).

    This agent:
    1. Loads documents from a directory
    2. Chunks and embeds them during initialization
    3. On query, embeds the query and finds similar chunks
    4. Returns the most relevant chunks as context
    """

    name: ClassVar[str] = "rag_research"
    system_prompt: ClassVar[str] = "Retrieves relevant context using semantic search."
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient | None = None,
        docs_path: str | Path = "docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        num_search_terms: int = 3,
        sub_agents: list[BaseAgent] | None = None,
        **kwargs
    ) -> None:
        """Initialize the RAG research agent.

        Args:
            embedding_client: Client for creating embeddings
            llm_client: Optional LLM client for query expansion
            docs_path: Path to directory containing documents
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            num_search_terms: Number of search terms to generate via LLM expansion
            sub_agents: Optional list of sub-agents
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if num_search_terms < 0:
            raise ValueError("num_search_terms must be non-negative")

        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.docs_path = Path(docs_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.num_search_terms = num_search_terms

        # Will be populated on first use
        self.chunks: list[str] = []
        self.chunk_sources: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._docs_loaded: bool = False

    def init_agent(self) -> None:
        """No-op: documents are loaded lazily on first use."""
        pass

    def _ensure_initialized(self) -> None:
        """Load and embed documents if not already done."""
        if not self._docs_loaded:
            self._load_and_embed_documents()
            self._docs_loaded = True

    def _load_and_embed_documents(self) -> None:
        """Load documents from docs_path and create embeddings.

        Uses cache if available and valid, otherwise generates and caches embeddings.
        """
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documents path not found: {self.docs_path}")

        # Try to load from cache
        if self._load_from_cache():
            self._log_status(f"Loaded embeddings from cache ({len(self.chunks)} chunks)")
            return

        # Collect all text files
        documents = []
        sources = []

        for file_path in self.docs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".txt", ".md", ".py"):
                content = file_path.read_text()
                documents.append(content)
                sources.append(str(file_path))

        if not documents:
            raise ValueError(f"No documents found in {self.docs_path}")

        # Chunk documents
        for doc, source in zip(documents, sources):
            doc_chunks = self._chunk_text(doc)
            self.chunks.extend(doc_chunks)
            self.chunk_sources.extend([source] * len(doc_chunks))

        # Create embeddings
        self._log_status(f"Generating embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.embedding_client.embed(self.chunks)

        # Save to cache
        self._save_to_cache()
        self._log_status(f"Embeddings cached to {self._get_cache_dir()}")

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a newline or space
            if end < len(text):
                # Look for newline first
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos + 1
                else:
                    # Fall back to space
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start + self.chunk_size // 2:
                        end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self.docs_path / ".cache"

    def _get_docs_mtime(self) -> float:
        """Get the latest modification time of all documentation files."""
        latest_mtime = 0.0
        for file_path in self.docs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".txt", ".md", ".py"):
                mtime = file_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
        return latest_mtime

    def _load_from_cache(self) -> bool:
        """Load embeddings from cache if valid.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_dir = self._get_cache_dir()
        embeddings_file = cache_dir / "embeddings.npy"
        metadata_file = cache_dir / "metadata.json"

        # Check if cache files exist
        if not embeddings_file.exists() or not metadata_file.exists():
            return False

        try:
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Check if docs have been modified since cache creation
            cached_mtime = metadata.get('docs_mtime', 0)
            current_mtime = self._get_docs_mtime()

            if current_mtime > cached_mtime:
                # Docs have been modified, cache is invalid
                return False
            if metadata.get('chunk_size') != self.chunk_size:
                return False
            if metadata.get('chunk_overlap') != self.chunk_overlap:
                return False

            # Load embeddings and metadata
            self.embeddings = np.load(embeddings_file)
            self.chunks = metadata['chunks']
            self.chunk_sources = metadata['sources']

            return True

        except Exception as e:
            # If anything goes wrong, just regenerate
            self._log_status(f"Warning: Failed to load cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save embeddings and metadata to cache."""
        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(exist_ok=True)

        embeddings_file = cache_dir / "embeddings.npy"
        metadata_file = cache_dir / "metadata.json"

        try:
            # Save embeddings
            np.save(embeddings_file, self.embeddings)

            # Save metadata
            metadata = {
                'chunks': self.chunks,
                'sources': self.chunk_sources,
                'docs_mtime': self._get_docs_mtime(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)

        except Exception as e:
            self._log_status(f"Warning: Failed to save cache: {e}")

    def _log_status(self, message: str) -> None:
        """Write RAG status messages to the agent log, not stdout."""
        if self.log_file:
            self.write_log(f"[{self.name}] {message}\n")

    def _search(self, query: str) -> list[tuple[str, str, float]]:
        """Search for chunks similar to query.

        Args:
            query: Search query

        Returns:
            List of (chunk, source, score) tuples
        """
        self._ensure_initialized()

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Embed query
        query_embedding = self.embedding_client.embed_one(query)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((
                self.chunks[idx],
                self.chunk_sources[idx],
                float(similarities[idx]),
            ))

        return results

    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and embeddings.

        Args:
            query: Query embedding (1D array)
            embeddings: Document embeddings (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Normalize embeddings
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / embedding_norms

        # Dot product gives cosine similarity for normalized vectors
        return np.dot(embeddings_normalized, query_norm)

    def _expand_query(self, query: str) -> list[str]:
        """Expand user query into better search terms.

        Converts imperative queries like "Create an atom and get its position" into
        documentation-style search terms like "Atom class", "get_position method".

        Args:
            query: Original user query

        Returns:
            List of search queries (original + expanded)
        """
        if not self.llm_client:
            # No LLM available, just use original query
            return [query]

        expansion_prompt = f"""Given this user request, extract {self.num_search_terms} key search terms that would help find relevant documentation.

User request: "{query}"

Return ONLY the search terms, one per line. Focus on:
- Class names, method names, API terms
- Key concepts and functionality
- Documentation-style language

Example:
User request: "Create an atom and get its position"
Output:
Atom class
get_position method
create atom instance"""

        try:
            response = self.llm_client.generate(
                prompt=expansion_prompt,
                system_prompt="You are a search query expansion assistant. Extract key documentation search terms."
            )

            # Parse response into lines
            expanded_terms = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Return original query + expanded terms
            return [query] + expanded_terms
        except Exception:
            # If expansion fails, fall back to original query
            return [query]

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by searching for relevant context.

        Args:
            state: Input state containing the user's query

        Returns:
            Dictionary with relevant context as a message
        """
        # Extract query
        if isinstance(state, str):
            query = state
        else:
            if state.messages:
                query = state.messages[-1].content
            else:
                query = ""

        if not query:
            response = "No query provided."
        else:
            # Expand query to get better search terms
            search_queries = self._expand_query(query)

            # Log the search queries being used
            if self.log_file:
                search_log = "Search queries:\n"
                for i, sq in enumerate(search_queries, 1):
                    search_log += f"  {i}. {sq}\n"
                self.write_log(search_log + "\n")

            # Search with all queries and aggregate results
            all_results = {}  # Use dict to deduplicate by chunk
            for search_query in search_queries:
                results = self._search(search_query)
                for chunk, source, score in results:
                    # Keep the highest score for each chunk
                    if chunk not in all_results or score > all_results[chunk][1]:
                        all_results[chunk] = (source, score)

            if all_results:
                # Sort by score and take top_k
                sorted_results = sorted(
                    all_results.items(),
                    key=lambda x: x[1][1],
                    reverse=True
                )[:self.top_k]

                context_parts = []
                for chunk, (source, score) in sorted_results:
                    context_parts.append(f"[Source: {source}, Score: {score:.3f}]\n{chunk}")
                response = "\n\n---\n\n".join(context_parts)
            else:
                response = "No relevant context found."

        response_message = Message(
            content=response,
            type="ai",
            additional_kwargs={"agent": "rag_research"}
        )
        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing relevant context
        """
        query = messages[-1].content if messages else ""
        state = State(messages=messages)
        result = self(state)
        return [result["message"]]

    def log_interaction(self, input_data: str, output_data: str) -> None:
        source_lines = [line for line in output_data.splitlines() if line.startswith("[Source:")]
        summary = "\n".join(source_lines) if source_lines else "(no results)"
        super().log_interaction(input_data, f"Retrieved:\n{summary}")

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override needed - uses embeddings directly."""
        return None
