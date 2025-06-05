from scholarqa import ScholarQA
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever

retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20)
paper_finder = PaperFinder(retriever, n_rerank=50, context_threshold=0.5)
scholar_qa = ScholarQA(paper_finder=paper_finder)

print(scholar_qa.answer_query("Which is the 9th planet in our solar system?"))
