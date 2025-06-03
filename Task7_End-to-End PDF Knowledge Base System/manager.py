from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

class ElasticsearchManager:
    def __init__(self, es_host: str = "http://localhost:9200", index_name: str = "pdf_knowledge_base"):
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        self.check_connection()

    def check_connection(self):
        """Checks if the Elasticsearch cluster is reachable."""
        try:
            if not self.es.ping():
                raise ValueError("Elasticsearch connection failed!")
            print(f"Connected to Elasticsearch: {self.es.info().body['cluster_name']}")
        except Exception as e:
            print(f"Could not connect to Elasticsearch: {e}")
            print("Please ensure Elasticsearch is running and accessible at the specified host.")
            exit(1) # Exit if connection fails

    def create_index(self):
        """Creates the Elasticsearch index with a specific mapping."""
        if self.es.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' already exists.")
            return

        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        },
                        "my_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "my_analyzer"},
                    "author": {"type": "keyword"},
                    "publication_date": {"type": "date"},
                    "full_text": {"type": "text", "analyzer": "my_analyzer"},
                    "summary": {"type": "text", "analyzer": "my_analyzer"},
                    "keywords": {"type": "keyword"},
                    "persons": {"type": "keyword"},
                    "organizations": {"type": "keyword"},
                    "locations": {"type": "keyword"},
                    "tables": {"type": "nested"}, # For structured data
                    "figures": {"type": "nested"}, # For structured data
                    "timestamp": {"type": "date"}
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=settings)
        print(f"Index '{self.index_name}' created successfully.")

    def index_document(self, doc_id: str, document: dict):
        """Indexes a single document into Elasticsearch."""
        try:
            self.es.index(index=self.index_name, id=doc_id, document=document)
            print(f"Document '{doc_id}' indexed successfully.")
        except Exception as e:
            print(f"Error indexing document '{doc_id}': {e}")

    def search_documents(self, query: str, fields: list = None, size: int = 10) -> list:
        """
        Performs a full-text search across specified fields or all text fields.
        Returns a list of matching documents.
        """
        if fields is None:
            # Default fields to search if not specified
            fields = ["title", "full_text", "summary", "keywords", "persons", "organizations", "locations"]

        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "fuzziness": "AUTO" # Basic typo tolerance
                }
            },
            "size": size
        }
        try:
            res = self.es.search(index=self.index_name, body=search_body)
            hits = []
            for hit in res['hits']['hits']:
                hits.append({"_id": hit['_id'], "_score": hit['_score'], **hit['_source']})
            return hits
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def semantic_search(self, query_vector: list, size: int = 10) -> list:
        """
        Placeholder for semantic search using vector similarity.
        Requires a vector field in the index and a way to generate query_vector.
        """
        print("Semantic search requires vector embeddings and a dedicated vector field in Elasticsearch.")
        print("This is a placeholder. Implement vector generation and K-NN search here.")
        return []

    def get_document_by_id(self, doc_id: str) -> dict or None:
        """Retrieves a document by its ID."""
        try:
            response = self.es.get(index=self.index_name, id=doc_id)
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            print(f"Error retrieving document '{doc_id}': {e}")
            return None