from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(["https://tce.edu/about/founder", "https://tce.edu/about/history"])
docs = loader.load()
for doc in docs:
    print(doc.metadata["source"])
    print(f"Total length: {len(doc.page_content)} characters")
    print(doc.page_content[-1500:])  # print the END of the page instead, where nav usually ends
    print("=" * 80)